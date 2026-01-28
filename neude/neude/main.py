import argparse
from neude import fuzzer
import multiprocessing


class PythonFuzz(object):
    def __init__(self, func):
        self.function = func

    def __call__(self, *args, **kwargs):
        parser = argparse.ArgumentParser(description='Coverage-guided fuzzer for python packages')
        parser.add_argument('dirs', type=str, nargs='*',
                            help="one or more directories/files to use as seed corpus. the first directory will be used to save the generated test-cases")
        parser.add_argument('--exact-artifact-path', type=str, help='set exact artifact path for crashes/ooms')
        parser.add_argument('--regression',
                            type=bool,
                            default=False,
                            help='run the fuzzer through set of files for regression or reproduction')
        parser.add_argument('--rss-limit-mb', type=int, default=1024*10, help='Memory usage in MB')
        parser.add_argument('--max-input-size', type=int, default=4096, help='Max input size in bytes')
        parser.add_argument('--dict', type=str, help='dictionary file')
        parser.add_argument('--close-fd-mask', type=int, default=0, help='Indicate output streams to close at startup')
        parser.add_argument('--runs', type=int, default=-1, help='Number of individual test runs, -1 (the default) to run indefinitely.')
        parser.add_argument('--timeout', type=int, default=600000,
                            help='If input takes longer then this timeout the process is treated as failure case')
        parser.add_argument('--cov-vec-save-iter', type=int, default=50000, help='Iterations to save coverage vector')
        parser.add_argument('--has-model', type=bool, default=False,help='to justfiy the system is tested use model')
        parser.add_argument('--use-nc', type=bool, default=False,help='use nc, else use pt')
        parser.add_argument('--batch-size', type=int, default=1, help='一次测试使用的种子数')
        args = parser.parse_args()
        f = fuzzer.Fuzzer(self.function, args.dirs, args.exact_artifact_path,
                          args.rss_limit_mb, args.timeout, args.regression, args.max_input_size,
                          args.close_fd_mask, args.runs, args.cov_vec_save_iter, args.has_model,
                          args.use_nc, args.batch_size, args.dict)
        f.start()


class Neude(object):
    def __init__(self, func):
        self.function = func

    def __call__(self, *args, **kwargs):
        parser = argparse.ArgumentParser(description='Coverage-guided fuzzer for python packages')
        parser.add_argument('dirs', type=str, nargs='*',
                            help="one or more directories/files to use as seed corpus. the first directory will be used to save the generated test-cases")
        parser.add_argument('--exact-artifact-path', type=str, help='set exact artifact path for crashes/ooms')
        parser.add_argument('--regression',
                            type=bool,
                            default=False,
                            help='run the fuzzer through set of files for regression or reproduction')
        parser.add_argument('--rss-limit-mb', type=int, default=1024*10, help='Memory usage in MB')
        parser.add_argument('--max-input-size', type=int, default=4096, help='Max input size in bytes')
        parser.add_argument('--dict', type=str, help='dictionary file')
        parser.add_argument('--close-fd-mask', type=int, default=0, help='Indicate output streams to close at startup')
        parser.add_argument('--runs', type=int, default=-1, help='Number of individual test runs, -1 (the default) to run indefinitely.')
        parser.add_argument('--timeout', type=int, default=600000,
                            help='If input takes longer then this timeout the process is treated as failure case')
        parser.add_argument('--cov-vec-save-iter', type=int, default=50000, help='Iterations to save coverage vector')
        parser.add_argument('--has-model', type=bool, default=False,help='to justfiy the system is tested use model')
        parser.add_argument('--use-nc', type=bool, default=False,help='use nc, else use pt')
        parser.add_argument('--batch-size', type=int, default=1, help='一次测试使用的种子数')
        args = parser.parse_args()
        f = fuzzer.Fuzzer(self.function, args.dirs, args.exact_artifact_path,
                          args.rss_limit_mb, args.timeout, args.regression, args.max_input_size,
                          args.close_fd_mask, args.runs, args.cov_vec_save_iter, args.has_model,
                          args.use_nc, args.batch_size, args.dict)
        f.start()


if __name__ == '__main__':
    # multiprocessing.set_start_method('spawn')
    # PythonFuzz()
    Neude()

