import numpy as np
import matplotlib.pyplot as plt

file_path = r"D:\02_02_26_Recordings\b210_capture_full_004_horizontal.iq"
samp_rate = 2e6  # 2 MHz

data = np.fromfile(file_path, dtype=np.complex64)

print(f"Loaded {len(data)} samples.")
print(f"Total duration: {len(data) / samp_rate:.2f} seconds")

def get_slice(data, start_sec, duration_sec, fs):
    start_idx = int(start_sec * fs)
    end_idx = int((start_sec + duration_sec) * fs)
    return data[start_idx:end_idx]

def split_chunks(samples):
    num_chunks = len(samples) // 1024
    trimmed_data = samples[:num_chunks * 1024]

    chunks = trimmed_data.reshape(-1, 1024)

    print(f"Original length: {len(bpsk_samples)}")
    print(f"Reshaped shape: {chunks.shape}")

    return chunks

start_time = 3.65

bpsk_samples = get_slice(data, start_sec=3.6, duration_sec=4, fs=samp_rate)
chunked_bpsk = split_chunks(bpsk_samples)

plt.figure(figsize=(12, 5))
plt.plot(np.real(chunked_bpsk[2000]))
plt.plot(np.imag(chunked_bpsk[2000]))
plt.title("BPSK Sample Chunk")
plt.show()

# re = np.real(bpsk_samples)
# im = np.imag(bpsk_samples)

# plt.figure(figsize=(12,5))
# plt.plot(re)
# plt.plot(im)
# plt.show()
