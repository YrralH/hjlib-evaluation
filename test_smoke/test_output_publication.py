'''Failure and no-clobber smoke for staged output-directory publication.'''
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from hjlib_systools.fileops import Publication_Destination_Exists_Error

from hjlib_evaluation.output_publication import staged_output_directory


def smoke_test_output_publication() -> None:
    with TemporaryDirectory() as root_value:
        root = Path(root_value)
        final = root / 'result'
        with staged_output_directory(final) as stage:
            (stage / 'value.txt').write_text('complete', encoding='utf-8')
        assert (final / 'value.txt').read_text(encoding='utf-8') == 'complete'
        try:
            with staged_output_directory(final):
                pass
        except FileExistsError:
            pass
        else:
            raise AssertionError('published output must not be overwritten')

        failed = root / 'failed'
        expected = RuntimeError('writer failed')
        try:
            with staged_output_directory(failed) as stage:
                (stage / 'partial.txt').write_text('partial', encoding='utf-8')
                raise expected
        except RuntimeError as error:
            assert error is expected
        else:
            raise AssertionError('writer failure must propagate')
        assert not failed.exists()
        assert not tuple(root.glob('.failed.partial-*'))

        raced = root / 'raced'
        retained_stage: Path | None = None
        try:
            with staged_output_directory(raced) as stage:
                retained_stage = stage
                (stage / 'source.txt').write_text('source', encoding='utf-8')
                raced.mkdir()
                (raced / 'destination.txt').write_text(
                    'destination', encoding='utf-8')
        except Publication_Destination_Exists_Error as error:
            assert retained_stage is not None
            assert error.staging == retained_stage
            assert (retained_stage / 'source.txt').is_file()
            assert (raced / 'destination.txt').is_file()
            shutil.rmtree(retained_stage)
        else:
            raise AssertionError('destination race must preserve both directories')


def test_output_publication() -> None:
    smoke_test_output_publication()


if __name__ == '__main__':
    smoke_test_output_publication()
