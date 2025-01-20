"""Utilities related to CP2K."""

from aiida_lsmo.utils import HARTREE2EV

# Functions to get inputs


def get_input_multiplicity(structure, protocol_settings):
    """ Compute the total multiplicity of the structure,
    by summing the atomic magnetizations:
    multiplicity = 1 + sum_i ( natoms_i * magnetization_i ), for each atom_type i
    """
    multiplicity = 1
    all_atoms = structure.get_ase().get_chemical_symbols()
    for key, value in protocol_settings['initial_magnetization'].items():
        multiplicity += all_atoms.count(key) * value
    multiplicity = int(round(multiplicity))
    multiplicity_dict = {'FORCE_EVAL': {'DFT': {'MULTIPLICITY': multiplicity}}}
    if multiplicity != 1:
        multiplicity_dict['FORCE_EVAL']['DFT']['UKS'] = True
    return multiplicity_dict


def get_kinds_section(structure, protocol_settings):
    """ Write the &KIND sections given the structure and the settings_dict"""
    kinds = []
    all_atoms = set(structure.get_ase().get_chemical_symbols())
    for atom in all_atoms:
        kinds.append({
            '_': atom,
            'BASIS_SET': protocol_settings['basis_set'][atom],
            'POTENTIAL': protocol_settings['pseudopotential'][atom],
            'MAGNETIZATION': protocol_settings['initial_magnetization'][atom],
        })
    return {'FORCE_EVAL': {'SUBSYS': {'KIND': kinds}}}


def get_kinds_with_ghost_section(structure, protocol_settings):
    """Write the &KIND sections given the structure and the settings_dict, and add also GHOST atoms"""
    kinds = []
    all_atoms = set(structure.get_ase().get_chemical_symbols())
    for atom in all_atoms:
        kinds.append({
            '_': atom,
            'BASIS_SET': protocol_settings['basis_set'][atom],
            'POTENTIAL': protocol_settings['pseudopotential'][atom],
            'MAGNETIZATION': protocol_settings['initial_magnetization'][atom],
        })
        kinds.append({'_': atom + "_ghost", 'BASIS_SET': protocol_settings['basis_set'][atom], 'GHOST': True})
    return {'FORCE_EVAL': {'SUBSYS': {'KIND': kinds}}}


def get_bsse_section(natoms_a, natoms_b, mult_a=1, mult_b=1, charge_a=0, charge_b=0):
    """Get the &FORCE_EVAL/&BSSE section."""
    bsse_section = {
        'FORCE_EVAL': {
            'BSSE' : {
                'FRAGMENT': [{
                'LIST': '1..{}'.format(natoms_a)
                },
                {
                'LIST': '{}..{}'.format(natoms_a + 1, natoms_a + natoms_b)
                }],
                'CONFIGURATION': [
                    { # A fragment with basis set A
                        'MULTIPLICITY': mult_a,
                        'CHARGE': charge_a,
                        'GLB_CONF': '1 0',
                        'SUB_CONF': '1 0',
                        },
                    { # B fragment with basis set B
                        'MULTIPLICITY': mult_b,
                        'CHARGE': charge_b,
                        'GLB_CONF': '0 1',
                        'SUB_CONF': '0 1',
                        },
                    { # A fragment with basis set A+B
                        'MULTIPLICITY': mult_a,
                        'CHARGE': charge_a,
                        'GLB_CONF': '1 1',
                        'SUB_CONF': '1 0',
                        },
                    { # B fragment with basis set A+B
                        'MULTIPLICITY': mult_b,
                        'CHARGE': charge_b,
                        'GLB_CONF': '1 1',
                        'SUB_CONF': '0 1',
                        },
                    { # A+B fragments with basis set A+B
                        'MULTIPLICITY': mult_a + mult_b - 1,
                        'CHARGE': charge_a + charge_b,
                        'GLB_CONF': '1 1',
                        'SUB_CONF': '1 1',
                        }
                ]
            }
        }
    }
    return bsse_section


# Functions to parse results


def ot_has_small_bandgap(cp2k_input, cp2k_output, bandgap_thr_ev):
    """ Returns True if the calculation used OT and had a smaller bandgap then the guess needed for the OT.
    (NOTE: It has been observed also negative bandgap with OT in CP2K!)
    cp2k_input: dict
    cp2k_output: dict
    bandgap_thr_ev: float [eV]
    """
    list_true = [True, 'T', 't', '.TRUE.', 'True', 'true']  #add more?
    try:
        ot_settings = cp2k_input['FORCE_EVAL']['DFT']['SCF']['OT']
        if '_' not in ot_settings.keys() or ot_settings['_'] in list_true:  #pylint: disable=simplifiable-if-statement
            using_ot = True
        else:
            using_ot = False
    except KeyError:
        using_ot = False
    min_bandgap_ev = min(cp2k_output["bandgap_spin1_au"], cp2k_output["bandgap_spin2_au"]) * HARTREE2EV
    is_bandgap_small = (min_bandgap_ev < bandgap_thr_ev)
    return using_ot and is_bandgap_small