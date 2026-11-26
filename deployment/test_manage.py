import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import manage as m

class DeploymentTests(unittest.TestCase):
    def test_memory_totals_and_unique_ports(self):
        self.assertEqual(len({x['port'] for x in m.MODELS}),9)
        self.assertNotIn(8006,{x['port'] for x in m.MODELS})
        for gpu in range(4): self.assertLessEqual(sum(x['memory'] for x in m.MODELS if x['gpu']==gpu),.900001)

    def test_build_command(self):
        model=m.model_for('mistral-small31')
        cmd=m.build_command(model,'/snapshot')
        self.assertIn('--language-model-only',cmd)
        self.assertEqual(cmd[cmd.index('--max-num-seqs')+1],'4')
        self.assertEqual(cmd[cmd.index('--load-format')+1],'mistral')
        self.assertNotIn('--quantization',cmd)

    def test_pid_identity_and_reuse(self):
        state={'pid':123,'start_ticks':'100','command_line':'expected'}
        with patch.object(m,'process',return_value={'pid':123,'uid':m.os.getuid(),'state':'S','start_ticks':'101','command':'expected'}):
            self.assertFalse(m.matches(state))
        with patch.object(m,'process',return_value={'pid':123,'uid':m.os.getuid(),'state':'S','start_ticks':'100','command':'expected'}):
            self.assertTrue(m.matches(state))

    def test_protected_existing_endpoint(self):
        with patch.object(m.os,'pidfd_open') as opened:
            with self.assertRaisesRegex(RuntimeError,'protected'): m.stop(m.model_for('qwen3-8b'))
            opened.assert_not_called()

    def test_stale_stop_never_signals(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(m,'ROOT',Path(directory)), patch.object(m,'matches',return_value=False), patch.object(m.os,'pidfd_open') as opened:
            (Path(directory)/'state').mkdir()
            with self.assertRaisesRegex(RuntimeError,'No verified'): m.stop(m.model_for('qwen3-4b'))
            opened.assert_not_called()

    def test_unmanaged_allocation_blocks_launch(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(m,'ROOT',Path(directory)), patch.object(m,'ancestor_owned',return_value=False), patch.object(m,'gpu_inventory',return_value={'gpus':[{'index':'0','uuid':'GPU-0'}],'processes':[{'gpu_uuid':'GPU-0','pid':'987'}]}), patch.object(m.subprocess,'Popen') as launch:
            root=Path(directory)
            (root/'state').mkdir()
            (root/'logs').mkdir()
            (root/'results').mkdir()
            m.save(root/'state/preflight.json',{'models':[{'key':'qwen3-4b','access':'available','revision':'abc'}]})
            m.save(root/'state/qwen3-4b-download.json',{'revision':'abc','snapshot':'/snapshot'})
            with patch.object(m.socket,'socket'):
                with self.assertRaisesRegex(RuntimeError,'Unmanaged'): m.start(m.model_for('qwen3-4b'))
            launch.assert_not_called()

    def test_endpoint_identity(self):
        with patch.object(m,'api',return_value={'body':{'data':[{'id':'wrong','max_model_len':4096}]}}):
            with self.assertRaisesRegex(RuntimeError,'identity'): m.identity(m.model_for('qwen3-4b'))

if __name__=='__main__': unittest.main()
