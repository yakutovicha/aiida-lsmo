from aiida.common import datastructures
from aiida.engine import CalcJob
from aiida.orm import RemoteData, Float

class VACCalculation(CalcJob):
    """AiiDA calculation plugin wrapping the MOFVacLevel executable."""
    _DEFAULT_INPUT_FILE = "vac_vacuum.py"
    _DEFAULT_OUTPUT_FILE = "log.out"
    _DEFAULT_PARENT_CALC_FLDR_NAME = "parent_calc/"
    @classmethod
    def define(cls, spec):
        """Define inputs and outputs of the calculation."""
        super(VACCalculation, cls).define(spec)

        # new ports
        spec.input('folder', valid_type=RemoteData, help='First file to be compared.')
        spec.output('vac_level', valid_type=Float, help='diff between file1 and file2.')

        spec.input('metadata.options.output_filename', valid_type=str, default='log.out' )
        spec.inputs['metadata']['options']['resources'].default = {
                                            'num_machines': 1,
                                            'num_mpiprocs_per_machine': 1,
                                            }
        spec.inputs['metadata']['options']['parser_name'].default = 'lsmo.vac_vaccum_parser'
        spec.exit_code(
            200,
            "ERROR_NO_RETRIEVED_FOLDER",
            message="The retrieved folder data node could not be accessed.",
        )
        spec.exit_code(300, 'ERROR_MISSING_OUTPUT_FILES',
                message='Calculation did not produce all expected output files.')
        spec.exit_code(
            301, "ERROR_OUTPUT_READ", message="The output file could not be read."
        )
        spec.exit_code(
            302, "ERROR_OUTPUT_PARSE", message="The output file could not be parsed."
        )


    def prepare_for_submission(self, folder):
        """
        Create input files.

        :param folder: an `aiida.common.folders.Folder` where the plugin should temporarily place all files needed by
            the calculation.
        :return: `aiida.common.datastructures.CalcInfo` instance
        """
        #lines= ['from mof_vac_level import MOFVacLevel', 'mvl = MOFVacLevel(\'parent_calc/aiida-v_hartree-1_0.cube\')', 'value = mvl.get_vacuum_potential(res=0.4, cube_size= [25, 25, 25] )','print(value)']
        lines = "from mof_vac_level import"
        with folder.open("vac_vacuum.py", 'w') as fobj:
        #with open(folder.get_abs_path(self._DEFAULT_INPUT_FILE), mode='w') as fobj:
            #fobj.write(f'from mof_vac_level import \n  ')
            fobj.write("from mof_vac_level import MOFVacLevel \nmvl = MOFVacLevel(\'parent_calc/aiida-v_hartree-1_0.cube\') \nvalue = mvl.get_vacuum_potential(res=0.5, cube_size= [25, 25, 25] ) \nprint(value)") 
    




        codeinfo = datastructures.CodeInfo()
        codeinfo.cmdline_params = [self._DEFAULT_INPUT_FILE]
        codeinfo.stin_name = self._DEFAULT_INPUT_FILE
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.stdout_name = self._DEFAULT_OUTPUT_FILE

        # Prepare a `CalcInfo` to be returned to the engine
        calcinfo = datastructures.CalcInfo()
        calcinfo.cmdline_params = codeinfo.cmdline_params
        calcinfo.stdin_name = self._DEFAULT_INPUT_FILE
        calcinfo.stdout_name = self._DEFAULT_OUTPUT_FILE
        
        calcinfo.codes_info = [codeinfo]
        #calcinfo.local_copy_list = [
        #    (self.inputs.folder.uuid, self.inputs.folder.filename, self.inputs.folder.filename)
        #]
        calcinfo.retrieve_list = [self.metadata.options.output_filename]
        
        #Make a link of cube_file folder 
        calcinfo.remote_symlink_list = []
        calcinfo.remote_copy_list = []
        
        comp_uuid = self.inputs.folder.computer.uuid
        remote_path = self.inputs.folder.get_remote_path()
        copy_info = (comp_uuid, remote_path, self._DEFAULT_PARENT_CALC_FLDR_NAME)
        # If running on the same computer - make a symlink.
        if self.inputs.code.computer.uuid == comp_uuid:
            calcinfo.remote_symlink_list.append(copy_info)
        # If not - copy the folder.
        else:
            calcinfo.remote_copy_list.append(copy_info)

        return calcinfo


