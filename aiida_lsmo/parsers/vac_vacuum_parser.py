from aiida.engine import ExitCode

from aiida.common import exceptions
from aiida.orm import SinglefileData, Float
from aiida.parsers.parser import Parser
from aiida.plugins import CalculationFactory

VACCalculation = CalculationFactory('lsmo.vac_vaccum')


class VACCalculationParser(Parser):
    
    def parse(self, **kwargs):
        """
        Parse outputs, store results in database.
        """
       

        

        output_filename = self.node.get_option('output_filename')
        
        files_retrieved = self.retrieved.list_object_names()
        files_expected = [output_filename]
        # Note: set(A) <= set(B) checks whether A is a subset of B
        if not set(files_expected) <= set(files_retrieved):
            self.logger.error(f"Found files '{files_retrieved}', expected to find '{files_expected}'")
            return self.exit_codes.ERROR_MISSING_OUTPUT_FILES
        # add output file
        self.logger.info(f"Parsing '{output_filename}'")
        with self.retrieved.open(output_filename, 'rb') as handle:
            value_file = float(handle.readline())
            output_node = Float(value_file)
        self.out('vac_level', output_node)

        return ExitCode(0)

