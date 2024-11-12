import numpy as np

def get_matrix_input(rows, cols, name="矩陣"):
    print(f"請輸入 {name} ({rows}x{cols})：")
    matrix = []
    for _ in range(rows):
        row = list(map(int, input().split()))
        if len(row) != cols:
            raise ValueError(f"每一行需要有 {cols} 個數字")
        matrix.append(row)
    return np.array(matrix)

def convolution_2d(input_matrix, kernel, padding=False, stride=1):
    input_rows, input_cols = input_matrix.shape
    kernel_size, _ = kernel.shape

    if padding:
        pad_size = kernel_size // 2
        input_matrix = np.pad(input_matrix, pad_size, mode='constant', constant_values=0)
    else:
        pad_size = 0

    output_rows = (input_rows + 2 * pad_size - kernel_size) // stride + 1
    output_cols = (input_cols + 2 * pad_size - kernel_size) // stride + 1

    output_matrix = np.zeros((output_rows, output_cols), dtype=int)

    for i in range(0, output_rows):
        for j in range(0, output_cols):
            region = input_matrix[i * stride:i * stride + kernel_size, j * stride:j * stride + kernel_size]
            output_matrix[i, j] = np.sum(region * kernel)

    return output_matrix

def main():
    input_rows, input_cols = map(int, input("輸入矩陣的行數和列數（例如：5 5）：").split())
    input_matrix = get_matrix_input(input_rows, input_cols, "輸入矩陣")

    kernel_size = int(input("輸入卷積核的大小（必須是奇數）："))
    if kernel_size % 2 == 0:
        raise ValueError("卷積核的大小必須是奇數")
    kernel = get_matrix_input(kernel_size, kernel_size, "卷積核")

    padding = input("是否使用 Padding (y/n)：").lower() == 'y'
    stride = int(input("步幅大小："))

    output_matrix = convolution_2d(input_matrix, kernel, padding, stride)
    print("卷積後的新矩陣：")
    for row in output_matrix:
        print(row)

if __name__ == "__main__":
    main()
