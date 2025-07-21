#!/usr/bin/env python3
import argparse, ast, xml.etree.ElementTree as ET
from tinygrad.helpers import unwrap
from ast import Lambda, Call, Attribute, Compare, Eq, NotEq, BoolOp, And, Or, Name, Constant, expr, arguments, arg
from typing import Callable

def parse_int(xml: ET.Element) -> int:
  return int(unwrap(xml.text), base=int(xml.get('Radix', 10)))

def parse_bool(val: ET.Element|str) -> bool:
  return {'true': True, 'false': False}[unwrap(val.text if isinstance(val, ET.Element) else val).lower()]

def parse_expression(xml: ET.Element) -> expr:
  if xml.tag == 'CondtionExpression':
    subtree = parse_expression(unwrap(xml.find('Expression')))
    return Lambda(args=arguments(posonlyargs=[], args=[arg(arg='kwargs')], kwonlyargs=[], kw_defaults=[], defaults=[]), body=subtree)
  assert xml.tag == 'Expression', xml
  match xml.get('Type'):
    case 'Literal':
      return Constant(value=parse_int(unwrap(xml.find('Value'))))
    case 'Id':
      return Call(func=Attribute(value=Name(id='kwargs'), attr='get') ,args=[Constant(value=unwrap(unwrap(xml.find('Label')).text)), Constant(value=None)], keywords=[])
    case 'Operator':
      cmpmap = {'==': Eq, '!=': NotEq}
      boolmap = {'&&': And, '||': Or}
      match unwrap(xml.find('Operator')).text:
        case '.fieldderef':
          value, identifier = unwrap(xml.find('Subexpressions'))[:2]
          return Call(func=Attribute(value=parse_expression(value), attr='get'), args=[parse_expression(identifier), Constant(value=None)], keywords=[])
        case op if op in cmpmap:
          left, right = unwrap(xml.find('Subexpressions'))[:2]
          return Compare(left=parse_expression(left), ops=[cmpmap[op]()], comparators=[parse_expression(right)])
        case op if op in boolmap:
          left, right = unwrap(xml.find('Subexpressions'))[:2]
          return BoolOp(op=boolmap[op](), values=[parse_expression(left), parse_expression(right)])
        case op: raise NotImplementedError(f'cannot turn operator {unwrap(op)} into python expression')
    case et: raise NotImplementedError(f'unknown expression type {et}')

from extra.rdna.isagen_quirks import fixup_instruction

def translate(xml_file:str) -> str:
  xml = ET.parse(xml_file).getroot()
  assert xml.tag == 'Spec', xml
  isa = unwrap(xml.find('ISA'))
  ret = []
  ret += ["from tinygrad.runtime.support.rdna_asm import Encoding, DataFormat, OperandType, InstructionOp, InstructionFlags, InstructionEncoding, InstructionOperand"]
  ret += [""]

  ret += ["# --- Encodings ---"]
  for enc in unwrap(isa.find('Encodings')):
    name = unwrap(unwrap(enc.find('EncodingName')).text)
    bitcount = parse_int(unwrap(enc.find('BitCount')))
    identifier_mask = parse_int(unwrap(enc.find('EncodingIdentifierMask')))
    ret += [f"{name} = Encoding("]
    ret += [f"  name={repr(name)},"]
    ret += [f"  bitcount={bitcount},"]
    ret += [ "  conditions={"]
    for condition in unwrap(enc.find('EncodingConditions')):
      condition_name = unwrap(unwrap(condition.find('ConditionName')).text)
      condition_expression_ast = parse_expression(unwrap(condition.find('CondtionExpression')))
      ret += [f"    {repr(condition_name)}: {ast.unparse(condition_expression_ast)},"]
    ret += [ "  },"]
    ret += [ "  bitfields={"]
    for bitfield in unwrap(enc.find('MicrocodeFormat/BitMap')):
      bitfield_name = unwrap(unwrap(bitfield.find('FieldName')).text)
      bitfield_start = parse_int(unwrap(bitfield.find('BitLayout/Range/BitOffset')))
      bitfield_size = parse_int(unwrap(bitfield.find('BitLayout/Range/BitCount')))
      ret += [f"    {repr(bitfield_name)}: {repr((bitfield_start, bitfield_start + bitfield_size - 1))},"]
    ret += [ "  },"]
    ret += [f"  identifier_mask=0b{bin(identifier_mask)[2:].zfill(bitcount)},"]
    ret += [ "  identifiers={"]
    for identifier in unwrap(enc.find('EncodingIdentifiers')):
      ret += [f"    0b{bin(parse_int(identifier))[2:].zfill(bitcount)},"]
    ret += [ "  },"]
    ret += [")"]
    ret += [""]

  ret += ["# --- DataFormats ---"]
  for df in unwrap(isa.find('DataFormats')):
    name = unwrap(unwrap(df.find('DataFormatName')).text)
    datatype = unwrap(unwrap(df.find('DataType')).text)
    bitcount = parse_int(unwrap(df.find('BitCount')))
    components = parse_int(unwrap(df.find('ComponentCount')))
    ret += [f"{name} = DataFormat({repr(name)}, {repr(datatype)}, {bitcount}, {components})"]
  ret += [""]

  ret += ["# --- OperandTypes ---"]
  for optps in unwrap(isa.find('OperandTypes')):
    name = unwrap(unwrap(optps.find('OperandTypeName')).text)
    predef: dict[str, int] = {}
    ret += [f"{name} = OperandType("]
    ret += [f"  name={repr(name)},"]
    if (pdvs:=optps.find('OperandPredefinedValues')) is None:
      ret += [ "  predef=None,"]
    else:
      ret += [ "  predef={"]
      for pdv in pdvs:
        pdv_name = unwrap(unwrap(pdv.find('Name')).text)
        pdv_value = parse_int(unwrap(pdv.find('Value')))
        ret += [f"    {repr(pdv_name)}: {pdv_value},"]
      ret += [ "  },"]
    if (bitmap:=optps.find('MicrocodeFormat/BitMap')) is None:
      ret += [ "  bitfields=None,"]
    else:
      ret += [ "  bitfields={"]
      for bitfield in bitmap:
        bitfield_name = unwrap(unwrap(bitfield.find('FieldName')).text)
        bitfield_start = parse_int(unwrap(bitfield.find('BitLayout/Range/BitOffset')))
        bitfield_size = parse_int(unwrap(bitfield.find('BitLayout/Range/BitCount')))
        ret += [f"    {repr(bitfield_name)}: {repr((bitfield_start, bitfield_start + bitfield_size - 1))},"]
      ret += [ "  },"]
    ret += [")"]
    ret += [""]

  ret += ["# --- Instructions ---"]
  for instr in unwrap(isa.find('Instructions')):
    fixup_instruction(instr)
    name = unwrap(unwrap(instr.find('InstructionName')).text)
    is_branch = parse_bool(unwrap(instr.find('InstructionFlags/IsBranch')))
    is_cbranch = parse_bool(unwrap(instr.find('InstructionFlags/IsConditionalBranch')))
    is_ibranch = parse_bool(unwrap(instr.find('InstructionFlags/IsIndirectBranch')))
    ret += [f"{name} = InstructionOp("]
    ret += [f"  name={repr(name)},"]
    ret += [ "  flags=InstructionFlags("]
    ret += [f"    is_branch={repr(is_branch)},"]
    ret += [f"    is_cbranch={repr(is_cbranch)},"]
    ret += [f"    is_ibranch={repr(is_ibranch)},"]
    ret += [ "  ),"]
    ret += [ "  encodings=["]
    for encoding in unwrap(instr.find('InstructionEncodings')):
      encoding_name = unwrap(unwrap(encoding.find('EncodingName')).text)
      encoding_condition = unwrap(unwrap(encoding.find('EncodingCondition')).text)
      encoding_opcode = parse_int(unwrap(encoding.find('Opcode')))
      ret += [ "    InstructionEncoding("]
      ret += [f"      encoding={encoding_name},"]
      ret += [f"      condition={repr(encoding_condition)},"]
      ret += [f"      opcode={encoding_opcode},"]
      ret += [ "      operands=["]
      for operand in unwrap(encoding.find('Operands')):
        operand_name = unwrap(fn.text) if (fn:=operand.find('FieldName')) is not None else None
        operand_dataformat = unwrap(unwrap(operand.find('DataFormatName')).text)
        operand_type = unwrap(unwrap(operand.find('OperandType')).text)
        operand_size = parse_int(unwrap(operand.find('OperandSize')))
        operand_input = parse_bool(unwrap(operand.get('Input')))
        operand_output = parse_bool(unwrap(operand.get('Output')))
        operand_implicit = parse_bool(unwrap(operand.get('IsImplicit')))
        ret += [ "        InstructionOperand("]
        ret += [f"          name={repr(operand_name)},"]
        ret += [f"          dataformat={operand_dataformat},"]
        ret += [f"          operand_type={operand_type},"]
        ret += [f"          size={operand_size},"]
        ret += [f"          input={operand_input},"]
        ret += [f"          output={operand_output},"]
        ret += [f"          implicit={operand_implicit},"]
        ret += [ "        ),"]
      ret += [ "      ],"]
      ret += [ "    ),"]
    ret += [ "  ],"]
    ret += [ ")"]
    ret += [""]

  return '\n'.join(ret)

def main():
  parser = argparse.ArgumentParser(prog='isagen', description="A tool to generate python bindings to amd's machine readable isa specifications")
  parser.add_argument('input')
  parser.add_argument('-o', '--output')
  args = parser.parse_args()
  with open(args.output, 'w+') as out_fd:
    out_fd.write(translate(args.input))

if __name__ == '__main__': main()
