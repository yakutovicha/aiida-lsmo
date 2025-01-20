# -*- coding: utf-8 -*-
"""Parsers for the specific usage of aiida-lsmo workchains."""

from aiida.common import OutputParsingError, NotExistent
from aiida.orm import Dict, BandsData
from aiida_cp2k.parsers import Cp2kBaseParser
from .parser_functions import parse_cp2k_output_bsse, parse_cp2k_output_advanced
from .vac_vacuum_parser import VACCalculationParser


class Cp2kBsseParser(Cp2kBaseParser):  # pylint: disable=too-few-public-methods
    """Advanced AiiDA parser class for a BSSE calculation in CP2K."""

    def _parse_stdout(self):
        """BSSE CP2K output file parser"""

        fname = self.node.get_attribute('output_filename')

        if fname not in self.retrieved.list_object_names():
            return self.exit_codes.ERROR_OUTPUT_STDOUT_MISSING

        try:
            output_string = self.retrieved.get_object_content(fname)
        except IOError:
            return self.exit_codes.ERROR_OUTPUT_STDOUT_READ

        result_dict = parse_cp2k_output_bsse(output_string)

        # nwarnings is the last thing to be printed in the CP2K output file:
        # if it is not there, CP2K didn't finish properly
        if 'nwarnings' not in result_dict:
            raise OutputParsingError('CP2K did not finish properly.')

        self.out('output_parameters', Dict(dict=result_dict))
        return None


class Cp2kAdvancedParser(Cp2kBaseParser):  # pylint: disable=too-few-public-methods
    """Advanced AiiDA parser class for the output of CP2K."""

    def _parse_stdout(self):
        """Advanced CP2K output file parser"""

        fname = self.node.get_attribute('output_filename')

        if fname not in self.retrieved.list_object_names():
            return self.exit_codes.ERROR_OUTPUT_STDOUT_MISSING

        try:
            output_string = self.retrieved.get_object_content(fname)
        except IOError:
            return self.exit_codes.ERROR_OUTPUT_STDOUT_READ

        result_dict = parse_cp2k_output_advanced(output_string)

        # nwarnings is the last thing to be printed in th eCP2K output file:
        # if it is not there, CP2K didn't finish properly
        if 'nwarnings' not in result_dict:
            raise OutputParsingError('CP2K did not finish properly.')

        # Compute the bandgap for Spin1 and Spin2 if eigen was parsed (works also with smearing!)
        if 'eigen_spin1_au' in result_dict:
            if result_dict['dft_type'] == 'RKS':
                result_dict['eigen_spin2_au'] = result_dict['eigen_spin1_au']

            lumo_spin1_idx = result_dict['init_nel_spin1']
            lumo_spin2_idx = result_dict['init_nel_spin2']
            if (lumo_spin1_idx > len(result_dict['eigen_spin1_au'])-1) or \
               (lumo_spin2_idx > len(result_dict['eigen_spin2_au'])-1):
                #electrons jumped from spin1 to spin2 (or opposite): assume last eigen is lumo
                lumo_spin1_idx = len(result_dict['eigen_spin1_au']) - 1
                lumo_spin2_idx = len(result_dict['eigen_spin2_au']) - 1
            homo_spin1 = result_dict['eigen_spin1_au'][lumo_spin1_idx - 1]
            homo_spin2 = result_dict['eigen_spin2_au'][lumo_spin2_idx - 1]
            lumo_spin1 = result_dict['eigen_spin1_au'][lumo_spin1_idx]
            lumo_spin2 = result_dict['eigen_spin2_au'][lumo_spin2_idx]
            result_dict['bandgap_spin1_au'] = lumo_spin1 - homo_spin1
            result_dict['bandgap_spin2_au'] = lumo_spin2 - homo_spin2

        if 'kpoint_data' in result_dict:
            bnds = BandsData()
            bnds.set_kpoints(result_dict['kpoint_data']['kpoints'])
            bnds.labels = result_dict['kpoint_data']['labels']
            bnds.set_bands(
                result_dict['kpoint_data']['bands'],
                units=result_dict['kpoint_data']['bands_unit'],
            )
            self.out('output_bands', bnds)
            del result_dict['kpoint_data']

        self.out('output_parameters', Dict(dict=result_dict))
        return None
