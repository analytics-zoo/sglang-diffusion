# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import multiprocessing as mp

import signal, os
from sglang.srt.utils import kill_process_tree
import uvicorn

from sglang.multimodal_gen.runtime.entrypoints.http_server import create_app
from sglang.multimodal_gen.runtime.managers.gpu_worker import run_scheduler_process
from sglang.multimodal_gen.runtime.server_args import ServerArgs, set_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import (
    configure_logger,
    logger,
    suppress_other_loggers,
)


def launch_server(server_args: ServerArgs, launch_http_server: bool = True):
    """
    Args:
        launch_http_server: False for offline local mode
    """
    configure_logger(server_args)
    suppress_other_loggers()

    # Start a new server with multiple worker processes
    logger.info("Starting server...")

    if True:  # Keep this check for internal code compatibility
        # Register the signal handler.
        # The child processes will send SIGQUIT to this process when any error happens
        # This process then clean up the whole process tree
        # Note: This sigquit handler is used in the launch phase, and may be replaced by
        # the running_phase_sigquit_handler in the tokenizer manager after the grpc server is launched.
        def launch_phase_sigquit_handler(signum, frame):
            logger.error(
                "Received sigquit from a child process. It usually means the child failed."
            )
            kill_process_tree(os.getpid())

        signal.signal(signal.SIGQUIT, launch_phase_sigquit_handler)

    # # Set mp start method
    mp.set_start_method("spawn", force=True)

    num_gpus = server_args.num_gpus
    processes = []

    # Pipes for master to talk to slaves
    task_pipes_to_slaves_w = []
    task_pipes_to_slaves_r = []
    for _ in range(num_gpus - 1):
        r, w = mp.Pipe(duplex=False)
        task_pipes_to_slaves_r.append(r)
        task_pipes_to_slaves_w.append(w)

    # Pipes for slaves to talk to master
    result_pipes_from_slaves_w = []
    result_pipes_from_slaves_r = []
    for _ in range(num_gpus - 1):
        r, w = mp.Pipe(duplex=False)
        result_pipes_from_slaves_r.append(r)
        result_pipes_from_slaves_w.append(w)

    # Launch all worker processes
    master_port = server_args.master_port or (server_args.master_port + 100)
    scheduler_pipe_readers = []
    scheduler_pipe_writers = []

    for i in range(num_gpus):
        reader, writer = mp.Pipe(duplex=False)
        scheduler_pipe_writers.append(writer)
        if i == 0:  # Master worker
            process = mp.Process(
                target=run_scheduler_process,
                args=(
                    i,  # local_rank
                    i,  # rank
                    master_port,
                    server_args,
                    writer,
                    None,  # No task pipe to read from master
                    None,  # No result pipe to write to master
                    task_pipes_to_slaves_w,
                    result_pipes_from_slaves_r,
                ),
                name=f"sglang-diffusionWorker-{i}",
                daemon=True,
            )
        else:  # Slave workers
            process = mp.Process(
                target=run_scheduler_process,
                args=(
                    i,  # local_rank
                    i,  # rank
                    master_port,
                    server_args,
                    writer,
                    None,  # No task pipe to read from master
                    None,  # No result pipe to write to master
                    task_pipes_to_slaves_r[i - 1],
                    result_pipes_from_slaves_w[i - 1],
                ),
                name=f"sglang-diffusionWorker-{i}",
                daemon=True,
            )
        scheduler_pipe_readers.append(reader)
        process.start()
        processes.append(process)

    # Wait for all workers to be ready
    scheduler_infos = []
    for writer in scheduler_pipe_writers:
        writer.close()

    # Close unused pipe ends in parent process
    for p in task_pipes_to_slaves_w:
        p.close()
    for p in task_pipes_to_slaves_r:
        p.close()
    for p in result_pipes_from_slaves_w:
        p.close()
    for p in result_pipes_from_slaves_r:
        p.close()

    for i, reader in enumerate(scheduler_pipe_readers):
        try:
            data = reader.recv()
        except EOFError:
            logger.error(
                f"Rank {i} scheduler is dead. Please check if there are relevant logs."
            )
            processes[i].join()
            logger.error(f"Exit code: {processes[i].exitcode}")
            raise

        if data["status"] != "ready":
            raise RuntimeError(
                "Initialization failed. Please see the error messages above."
            )
        scheduler_infos.append(data)
        reader.close()

    logger.debug("All workers are ready")

    if launch_http_server:
        logger.info("Starting FastAPI server.")

        # set for endpoints to access global_server_args
        set_global_server_args(server_args)

        app = create_app(server_args)
        uvicorn.run(
            app,
            log_config=None,
            log_level=server_args.log_level,
            host=server_args.host,
            port=server_args.port,
            reload=False,
        )
