import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('entrypoint', ROOT / 'run.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class EntrypointTests(unittest.TestCase):
    def test_help_without_gpu(self):
        result = subprocess.run([sys.executable, str(ROOT / 'run.py'), '--help'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('evaluate', result.stdout)

    def test_relative_paths(self):
        config = {key: './data' for key in entry.PATH_KEYS}
        with tempfile.TemporaryDirectory() as folder:
            result = entry.resolve_paths(config, {}, Path(folder))
            self.assertEqual(result['lmdb_dir'], str((Path(folder) / 'data').resolve()))
        self.assertEqual(config['lmdb_dir'], './data')

    def test_unknown_path_rejected(self):
        with self.assertRaises(ValueError):
            entry.resolve_paths({}, {'typo': '/data'}, ROOT)

    def test_missing_data_fails_before_training(self):
        config = {key: str(ROOT / 'nonexistent_test_data') for key in entry.PATH_KEYS}
        config.update(test_dataset=['Celeb-DF-v1'], lmdb=True)
        with self.assertRaises(FileNotFoundError):
            entry.check_data(config)

    def test_no_personal_paths_in_release_sources(self):
        for path in ROOT.rglob('*'):
            if path.suffix not in ('.py', '.yaml', '.json') or 'tests' in path.parts:
                continue
            text = path.read_text(encoding='utf-8-sig')
            self.assertNotIn('/home/' + 'dingyu', text, str(path))
            self.assertNotIn('/data2/' + 'dingyu', text, str(path))


if __name__ == '__main__':
    unittest.main()
