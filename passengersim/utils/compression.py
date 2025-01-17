import os.path


def compress_file(input_file: str, chunk_size: int = 65536):
    """Compresses a file using LZ4."""
    import lz4.frame

    output_file = str(input_file) + ".lz4"
    with open(input_file, "rb") as infile, open(output_file, "wb") as outfile:
        compressor = lz4.frame.LZ4FrameCompressor()
        outfile.write(compressor.begin())

        while True:
            chunk = infile.read(chunk_size)
            if not chunk:
                break
            compressed_chunk = compressor.compress(chunk)
            outfile.write(compressed_chunk)

        outfile.write(compressor.flush())


def decompress_file(input_file: str, chunk_size: int = 65536):
    """Decompresses a file using LZ4."""
    import lz4.frame

    input_file = str(input_file)
    if not input_file.endswith(".lz4"):
        raise ValueError("Input file must have a .lz4 extension")

    output_file = input_file[:-4]
    if os.path.exists(output_file):
        raise FileExistsError(f"Output file {output_file} already exists")
    with open(input_file, "rb") as infile, open(output_file, "wb") as outfile:
        decompressor = lz4.frame.LZ4FrameDecompressor()
        while True:
            chunk = infile.read(chunk_size)
            if not chunk:
                break
            decompressed_chunk = decompressor.decompress(chunk)
            outfile.write(decompressed_chunk)


def smart_open(filename: str, mode: str = "r"):
    """Opens a file with a compression filter based on the extension."""
    filename = str(filename)
    if filename.endswith(".lz4"):
        import lz4.frame

        return lz4.frame.open(filename, mode)
    elif filename.endswith(".gz"):
        import gzip

        return gzip.open(filename, mode)
    return open(filename, mode)
