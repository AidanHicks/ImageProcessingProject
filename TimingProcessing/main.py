import pandas as pd
import matplotlib.pyplot as plt

#Load Timing Files
base = pd.read_csv(
    "C:\\Users\\MESH USER\\Desktop\\MainProj\\Algo1\\Timings\\BaseTimings.csv",
    names=["iteration", "duration_ns"]
)

omp = pd.read_csv(
    "C:\\Users\\MESH USER\\Desktop\\MainProj\\Algo1\\Timings\\OpenMPTimings.csv",
    names=["iteration", "duration_ns"]
)

cuda = pd.read_csv(
    "C:\\Users\\MESH USER\\Desktop\\MainProj\\Algo1\\Timings\\CudaTimings.csv",
    names=["iteration", "duration_ns"]
)

#Downsample
base_down = base.iloc[::500]
omp_down  = omp.iloc[::500]
cuda_down = cuda.iloc[::500]

#Y-axis max values
base_max = base_down["duration_ns"].max()
omp_max  = omp_down["duration_ns"].max()
cuda_max = cuda_down["duration_ns"].max()

#Global max for comparison chart
global_max = max(base_max, omp_max, cuda_max)

#BaseAlgo - Line
plt.figure(figsize=(14, 6))
plt.plot(base_down["iteration"], base_down["duration_ns"])
plt.title("Base Algorithm Execution Time (Line Graph)")
plt.xlabel("Iteration")
plt.ylabel("Duration (nanoseconds)")
plt.grid(True)
plt.ylim(0, base_max)
plt.show()

#BaseAlgo - Bar
plt.figure(figsize=(16, 6))
plt.bar(base_down["iteration"], base_down["duration_ns"], width=400)
plt.title("Base Algorithm Execution Time (Bar Chart)")
plt.xlabel("Iteration")
plt.ylabel("Duration (nanoseconds)")
plt.grid(True, axis="y")
plt.ylim(0, base_max)
plt.show()

#OpenMP - Line
plt.figure(figsize=(14, 6))
plt.plot(omp_down["iteration"], omp_down["duration_ns"])
plt.title("OpenMP Execution Time (Line Graph)")
plt.xlabel("Iteration")
plt.ylabel("Duration (nanoseconds)")
plt.grid(True)
plt.ylim(0, omp_max)
plt.show()

#OpenMP - Bar
plt.figure(figsize=(16, 6))
plt.bar(omp_down["iteration"], omp_down["duration_ns"], width=400)
plt.title("OpenMP Execution Time (Bar Chart)")
plt.xlabel("Iteration")
plt.ylabel("Duration (nanoseconds)")
plt.grid(True, axis="y")
plt.ylim(0, omp_max)
plt.show()

#CUDA - Line
plt.figure(figsize=(14, 6))
plt.plot(cuda_down["iteration"], cuda_down["duration_ns"], color='green')
plt.title("CUDA Execution Time (Line Graph)")
plt.xlabel("Iteration")
plt.ylabel("Duration (nanoseconds)")
plt.grid(True)
plt.ylim(0, cuda_max)
plt.show()

#CUDA - Bar
plt.figure(figsize=(16, 6))
plt.bar(cuda_down["iteration"], cuda_down["duration_ns"], width=400, color='green')
plt.title("CUDA Execution Time (Bar Chart)")
plt.xlabel("Iteration")
plt.ylabel("Duration (nanoseconds)")
plt.grid(True, axis="y")
plt.ylim(0, cuda_max)
plt.show()

#Overall Comparison
plt.figure(figsize=(14, 6))
plt.plot(base_down["iteration"], base_down["duration_ns"], label="Base CPU")
plt.plot(omp_down["iteration"],  omp_down["duration_ns"], label="OpenMP")
plt.plot(cuda_down["iteration"], cuda_down["duration_ns"], label="CUDA", color="green")

plt.title("Execution Time Comparison (Base vs OpenMP vs CUDA)")
plt.xlabel("Iteration")
plt.ylabel("Duration (nanoseconds)")
plt.grid(True)
plt.ylim(0, global_max)
plt.legend()
plt.show()
