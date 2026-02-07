from unittest.mock import MagicMock, patch

from batch_process import batch_process


def test_batch_process():
    """Verify batch_process correctly launches subprocesses."""
    with patch("subprocess.Popen") as mock_popen:
        mock_popen.return_value = MagicMock()

        # Test with a small range: Jan and Feb 2023
        # End date is inclusive in pd.date_range
        processes = batch_process(
            start_date="2023-01-01", end_date="2023-02-01", freq="MS"
        )

        # 2023-01-01 and 2023-02-01
        assert len(processes) == 2
        assert mock_popen.call_count == 2

        # Check first call
        args, kwargs = mock_popen.call_args_list[0]
        cmd = args[0]
        assert cmd[0] == "python"
        assert "./dust.py" in cmd
        assert "-s" in cmd
        assert "2023-01-01" in cmd
        assert "-e" in cmd
        assert "2023-02-01" in cmd  # 2023-01-01 + 1 month
        assert "-o" in cmd
        assert "data/20230101.dat" in cmd

        # Check second call
        args, kwargs = mock_popen.call_args_list[1]
        cmd = args[0]
        assert "2023-02-01" in cmd
        assert "2023-03-01" in cmd  # 2023-02-01 + 1 month
        assert "data/20230201.dat" in cmd
