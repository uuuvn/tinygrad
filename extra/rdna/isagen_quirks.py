# Handcoded fixes for amd's machine readable gpu isa definitions
import xml.etree.ElementTree as ET
from tinygrad.helpers import unwrap

def fixup_instruction(xml: ET.Element):
  name = unwrap(unwrap(xml.find('InstructionName')).text)
  group = unwrap(unwrap(xml.find('FunctionalGroup/Name')).text)
  subgroup = unwrap(sg.text) if (sg:=xml.find('FunctionalGroup/Subgroup')) is not None else None
  if (group, subgroup) == ('VMEM', 'FLAT') and 'ATOMIC' not in name and name.startswith('GLOBAL_'):
    encodings = unwrap(xml.find('InstructionEncodings'))
    unwrap(unwrap(encodings[0].find('Operands'))[0].find('OperandSize')).text = '32'
