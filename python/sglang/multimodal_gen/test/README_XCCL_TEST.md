# XCCL Communication Test Suite

This test suite is designed to verify XCCL communication primitives on Intel XPU.

## Files

- `test_xccl_communication.py`: Main test script with 9 tests
- `run_xccl_test.sh`: Bash script to run the tests with proper environment variables

## Tests Included

1. **Basic Tensor Creation**: Verify XPU tensor creation
2. **Barrier**: Test synchronization across processes
3. **Broadcast**: Test broadcasting data from rank 0 to all ranks
4. **AllReduce**: Test sum reduction across all ranks
5. **Send/Recv**: Test point-to-point synchronous communication
6. **Async Send/Recv**: Test point-to-point asynchronous communication (isend/irecv)
7. **All-to-All**: Test collective all-to-all operation
8. **All-to-All-Single**: Test single-tensor all-to-all operation
9. **Manual All-to-All**: Test manual implementation using isend/irecv

## How to Run

### Method 1: Using the bash script (recommended)

```bash
cd /home/intel/xiangyu/study/sglang/python/sglang/multimodal_gen/test
chmod +x run_xccl_test.sh
./run_xccl_test.sh
```

### Method 2: Direct torchrun

```bash
cd /home/intel/xiangyu/study/sglang/python/sglang/multimodal_gen/test

# Set environment variables (optional but recommended for debugging)
export CCL_LOG_LEVEL=info
export CCL_ATL_TRANSPORT=ofi
export FI_PROVIDER=tcp

# Run the test
torchrun --nproc_per_node=4 --master_port=29500 test_xccl_communication.py
```

## Environment Variables

### CCL Configuration

- `CCL_LOG_LEVEL`: Set to `info`, `debug`, or `trace` for detailed logs
- `CCL_ATL_TRANSPORT`: Transport layer, usually `ofi` (libfabric) or `mpi`
- `FI_PROVIDER`: Fabric provider, options include:
  - `tcp`: TCP/IP transport (works everywhere, slower)
  - `verbs`: InfiniBand verbs (fastest, requires IB hardware)
  - `sockets`: Sockets provider
  - `psm2`: Intel OPA fabric

### Other Useful Variables

- `CCL_WORKER_COUNT`: Number of worker threads (default: auto)
- `CCL_WORKER_AFFINITY`: CPU affinity for workers
- `I_MPI_DEBUG`: MPI debug level (if using MPI transport)

## Expected Output

The test will run all 9 tests sequentially. For each test, you should see:

```
==================================================
Test X: [Test Name]
[Rank 0] Test description and progress
[Rank 1] Test description and progress
...
✓ [Test Name] completed successfully
```

At the end, a summary will be printed:

```
==================================================
TEST SUMMARY
==================================================
✓ PASS: Basic Tensor Creation
✓ PASS: Barrier
...
```

## Troubleshooting

### If a test hangs:

1. **Check which test is hanging**: The last log message will indicate where it stopped
2. **Enable debug logs**: Set `CCL_LOG_LEVEL=debug` or `CCL_LOG_LEVEL=trace`
3. **Try different transport**: Change `FI_PROVIDER` to `tcp` or `sockets`
4. **Check network**: Ensure all nodes can communicate with each other

### Common Issues:

1. **"No backend type associated with device type xpu"**
   - XCCL backend is not properly installed or registered
   - Check PyTorch XPU installation

2. **Timeout/Hang in collective operations**
   - Network connectivity issues
   - Incorrect process group initialization
   - XCCL bug or missing feature

3. **"Connection refused" errors**
   - Port conflicts (try changing `--master_port`)
   - Firewall blocking communication

## Interpreting Results

- **If Test 1-2 fail**: Basic setup issue (PyTorch XPU not working)
- **If Test 3-4 fail**: Collective operations not working
- **If Test 5-6 fail**: Point-to-point operations not working
- **If Test 7-8 fail**: All-to-all operations not supported by XCCL
- **If Test 9 fails**: Even manual implementation doesn't work (serious issue)

Based on which tests fail, we can determine:
- Use simpler parallelization strategies (avoid Ulysses if all-to-all doesn't work)
- Use manual implementations for unsupported operations
- Report bugs to oneCCL/PyTorch XPU teams
