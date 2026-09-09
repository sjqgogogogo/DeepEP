"""CPU checks for build configuration; these do not compile CUDA kernels."""

import contextlib
import io
import os
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

import torch.utils.cpp_extension


class LegacyBuildTest(unittest.TestCase):
    def test_build_modes(self):
        repo = Path(__file__).resolve().parents[1]
        for mode in (None, '0', '1'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / 'lib').mkdir()
                (root / 'lib/libnvshmem_host.so.3').touch()
                # Legacy-only builds must not try to resolve this library.
                if mode != '1':
                    (root / 'lib/libnccl.so.2').touch()
                environment = {
                    'EP_NVSHMEM_ROOT_DIR': directory,
                    'EP_NCCL_ROOT_DIR': directory,
                    'TORCH_CUDA_ARCH_LIST': '9.0',
                }
                if mode is not None:
                    environment['EP_LEGACY_ONLY'] = mode
                with patch.dict(os.environ, environment, clear=True), \
                        patch.object(torch.utils.cpp_extension, 'CUDAExtension') as extension, \
                        patch('setuptools.setup') as setup, contextlib.redirect_stdout(io.StringIO()):
                    namespace = runpy.run_path(str(repo / 'setup.py'), run_name='__main__')

                config = extension.call_args.kwargs
                legacy_only = mode == '1'
                self.assertEqual('csrc/kernels/backend/nccl.cu' in config['sources'], not legacy_only)
                self.assertEqual('-l:libnccl.so.2' in config['extra_link_args'], not legacy_only)
                for language in ('cxx', 'nvcc'):
                    self.assertEqual('-DEP_LEGACY_ONLY' in config['extra_compile_args'][language], legacy_only)
                self.assertIn('csrc/kernels/legacy/internode_ll.cu', config['sources'])
                self.assertIn('-l:libnvshmem_host.so.3', config['extra_link_args'])
                self.assertIn(str(root / 'include'), config['include_dirs'])
                self.assertIn('EP_LEGACY_ONLY', namespace['persistent_env_names'])
                self.assertEqual(setup.call_args.kwargs['name'], 'deep_ep')


if __name__ == '__main__':
    unittest.main()
