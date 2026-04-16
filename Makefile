all:
	nvcc -O3 -lineinfo matrix_product.cu -o matrix_product
clean:
	rm -f matrix_product