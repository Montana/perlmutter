# Comprehensive Guide to Running MPI Applications on NERSC Perlmutter Supercomputer

## Introduction

This document provides detailed instructions for running Python-based MPI (Message Passing Interface) applications on the NERSC Perlmutter supercomputer. Perlmutter is a HPE Cray EX system featuring AMD EPYC processors and NVIDIA A100 GPUs, designed for large-scale scientific computing tasks.

## System Overview

Perlmutter consists of:
- CPU Partition: 3,072 nodes with AMD EPYC "Milan" processors (64 cores/node)
- GPU Partition: 1,792 nodes with AMD EPYC CPUs and 4 NVIDIA A100 GPUs per node
- Slingshot 11 interconnect with dragonfly topology
- 44 PB all-flash Lustre scratch file system

## SLURM Job Configuration

SLURM (Simple Linux Utility for Resource Management) manages workloads on Perlmutter. Below is a template for a SLURM job script that runs a Python MPI application:

```bash
#!/bin/bash
#SBATCH --job-name=mpi_test           # Name of the job
#SBATCH --account=your_account_here   # Allocation account to charge
#SBATCH --partition=regular           # Partition/queue to submit to
#SBATCH --nodes=2                     # Number of compute nodes
#SBATCH --ntasks-per-node=4           # MPI tasks per node (total: 8 tasks)
#SBATCH --cpus-per-task=16            # CPU cores per MPI task (for hybrid MPI+OpenMP)
#SBATCH --time=00:10:00               # Maximum runtime (HH:MM:SS)
#SBATCH --output=mpi_test_%j.out      # Output file (%j is replaced by job ID)
#SBATCH --error=mpi_test_%j.err       # Error file (%j is replaced by job ID)
#SBATCH --mail-type=BEGIN,END,FAIL    # Email notifications
#SBATCH --mail-user=user@example.com  # Email address
#SBATCH --constraint=cpu              # Use CPU nodes (remove for GPU nodes)
#SBATCH --qos=regular                 # Quality of service

echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on $SLURM_NNODES nodes: $SLURM_NODELIST"
echo "Using $SLURM_NTASKS tasks and $SLURM_CPUS_PER_TASK CPUs per task"

module purge
module load cpu                # For CPU nodes (use 'gpu' for GPU nodes)
module load PrgEnv-gnu         # Programming environment with GNU compilers
module load python
module load cray-mpich
module load cray-hdf5-parallel

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export MPICH_GPU_SUPPORT_ENABLED=1  # For GPU-aware MPI

source mpi_env

source activate mpi_env

srun --cpu-bind=cores python mpi_script.py

echo "Job finished at: $(date)"
```

## SLURM Directives Explained

| Directive | Description |
|:----------|:------------|
| `--job-name=mpi_test` | Sets a name for the job for identification in SLURM queue |
| `--account=your_account_here` | Specifies which project to charge compute time to |
| `--partition=regular` | Submits job to the regular queue (other options: shared, debug, etc.) |
| `--nodes=2` | Requests 2 compute nodes for the job |
| `--ntasks-per-node=4` | Runs 4 MPI tasks per node (8 total processes across 2 nodes) |
| `--cpus-per-task=16` | Allocates 16 CPU cores to each MPI task (for hybrid MPI+OpenMP) |
| `--time=00:10:00` | Sets maximum job runtime to 10 minutes (format: HH:MM:SS) |
| `--output=mpi_test_%j.out` | Redirects stdout to file with job ID appended |
| `--error=mpi_test_%j.err` | Redirects stderr to file with job ID appended |
| `--mail-type=BEGIN,END,FAIL` | Sends email notifications at job start, end, and failure |
| `--mail-user=user@example.com` | Email address for notifications |
| `--constraint=cpu` | Specifies to use CPU nodes (use 'gpu' for GPU nodes) |
| `--qos=regular` | Quality of service level (options: regular, premium, etc.) |

## Python MPI Script Example

Here's an example MPI Python script (`mpi_script.py`) using mpi4py:

```python
#!/usr/bin/env python
from mpi4py import MPI
import numpy as np
import time
import socket
import os

def main():
    # Initialize MPI environment
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    hostname = socket.gethostname()
    
    if rank == 0:
        print(f"Running MPI job with {size} processes")
        print(f"OMP_NUM_THREADS = {os.environ.get('OMP_NUM_THREADS', 'not set')}")
        print("Process distribution:")
    
    hostnames = comm.gather(hostname, root=0)
    
    if rank == 0:
        host_counts = {}
        for host in hostnames:
            if host in host_counts:
                host_counts[host] += 1
            else:
                host_counts[host] = 1
        
        for host, count in host_counts.items():
            print(f"  {host}: {count} processes")
    
    comm.Barrier()
    
    local_data = np.random.rand(1000000) * rank
    local_sum = np.sum(local_data)
    
    global_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)
    
    if rank == 0:
        print(f"Global sum: {global_sum}")
    
    comm.Barrier()
    
    if rank == 0:
        print("MPI job completed successfully")

if __name__ == "__main__":
    start_time = time.time()
    main()
    if MPI.COMM_WORLD.Get_rank() == 0:
        end_time = time.time()
        print(f"Total execution time: {end_time - start_time:.2f} seconds")
```

## Setup for MPI Environment

To create and configure a suitable Python environment for MPI applications:

1. Load the required modules:
   ```bash
   module load python
   module load cray-mpich
   ```

2. Create a new conda environment:
   ```bash
   conda create -n mpi_env python=3.9
   ```

3. Activate the environment:
   ```bash
   source activate mpi_env
   ```

4. Install required packages:
   ```bash
   MPICC=cc pip install mpi4py
   pip install numpy scipy matplotlib h5py
   ```

## Running on GPU Nodes

To utilize GPU acceleration on Perlmutter, modify your SLURM script:

```bash
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=4  # Request all 4 GPUs per node
```

And add to your environment setup:

```bash
module load gpu
module load cudatoolkit
export MPICH_GPU_SUPPORT_ENABLED=1
```

For CUDA-aware MPI with Python, ensure your script properly utilizes GPU resources:

```python
import cupy as cp  # CUDA-accelerated NumPy-like package

local_data = cp.random.rand(1000000) * rank
local_sum = cp.sum(local_data).get()  # Convert back to CPU for MPI
```

## Performance Optimization Tips

1. **Process Binding**: Use `--cpu-bind=cores` with srun to bind processes to specific cores
2. **Memory Placement**: Set `export MPICH_MEMORY_REPORT=1` to get memory usage reports
3. **Network Tuning**: Adjust `MPICH_OFI_NIC_POLICY=NUMA` to optimize network performance
4. **I/O Optimization**: Use parallel HDF5 for better I/O performance
5. **Load Balancing**: Distribute workloads evenly across MPI ranks

## Job Submission and Monitoring

Submit your job to the queue:

```bash
sbatch mpi_job.slurm
```

Check job status:

```bash
squeue -u $USER
```

View detailed job information:

```bash
scontrol show job <jobid>
```

Check job efficiency after completion:

```bash
sacct -j <jobid> --format=JobID,JobName,MaxRSS,Elapsed
```

## Debugging MPI Applications

For debugging MPI applications, add to your SLURM script:

```bash
#SBATCH --nodes=1  # Use fewer nodes for debugging
#SBATCH --ntasks=4
#SBATCH --time=00:30:00
#SBATCH --partition=debug

export MPICH_DEBUG_RANKS=1
export MPICH_ENV_DISPLAY=1

srun gdb -ex run --args python mpi_script.py
srun ddt --connect python mpi_script.py
```

## Best Practices

1. Start with small test runs before scaling up
2. Always set a reasonable time limit
3. Use the debug partition for testing
4. Monitor resource usage with Slurm tools
5. Keep checkpoint files for long-running jobs
6. Optimize I/O patterns to minimize file system contention
7. Scale tests to verify performance improvement with node count

## Additional Resources

- [NERSC Perlmutter Documentation](https://docs.nersc.gov/systems/perlmutter/)
- [Slurm Workload Manager Documentation](https://slurm.schedmd.com/)
- [MPI for Python Documentation](https://mpi4py.readthedocs.io/)
- [NERSC JupyterHub](https://jupyter.nersc.gov/) for interactive development
- [NERSC User Support](https://help.nersc.gov/) for additional assistance

## Troubleshooting Common Issues

1. **Job fails immediately**: Check resources requested vs. available
2. **MPI errors**: Verify module loads and environment settings
3. **Out of memory**: Reduce problem size or request more nodes
4. **Slow performance**: Check process binding and MPI communication patterns
5. **File system issues**: Use $SCRATCH for temporary files, not $HOME
