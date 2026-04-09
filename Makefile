all:
	nvcc matrix_product.cu -o matrix_product
clean:
	rm -f matrix_product