import base64
import os.path
from binascii import Error as BinAsciiError
from io import StringIO
from typing import Any


def compress_file(input_file: str | os.PathLike[str], chunk_size: int = 65536, rm_original: bool = False):
    """Compresses a file using LZ4."""
    import lz4.frame

    output_file = os.fspath(input_file) + ".lz4"
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
    if rm_original:
        os.remove(input_file)


def decompress_file(input_file: str | os.PathLike[str], chunk_size: int = 65536):
    """Decompresses a file using LZ4."""
    import lz4.frame

    input_file = os.fspath(input_file)
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


def smart_open(filename: str | os.PathLike, mode: str = "r"):
    """Opens a file with a compression filter based on the extension."""
    filename = os.fspath(filename)
    if filename.endswith(".lz4"):
        import lz4.frame

        return lz4.frame.open(filename, mode)
    elif filename.endswith(".gz"):
        import gzip

        return gzip.open(filename, mode)
    return open(filename, mode)


def serialize_to_file(filename: str | os.PathLike, obj):
    """Dumps an object to a file."""

    import pickle

    with smart_open(filename, "wb") as f:
        pickle.dump(obj, f)


def deserialize_from_file(filename: str | os.PathLike) -> Any:
    """Loads an object from a file."""

    import pickle

    with smart_open(filename, "rb") as f:
        return pickle.load(f)


def serialize_to_bytes(obj: Any, b64: bool = True) -> bytes:
    import dill as pickle
    import lz4.frame

    serialized_value = pickle.dumps(obj)
    compressed_value = lz4.frame.compress(serialized_value)
    if b64:
        compressed_value = base64.b64encode(compressed_value)
    return compressed_value


def deserialize_from_bytes(compressed_content: bytes, b64: bool = True) -> Any:
    import dill as pickle
    import lz4.frame

    if b64:
        # attempt to b64 decode, but if there are errors assume it wasn't base64
        try:
            compressed_content = base64.b64decode(compressed_content, validate=True)
        except BinAsciiError:
            pass
    serialized_value = lz4.frame.decompress(compressed_content)
    return pickle.loads(serialized_value)


def pretty_pack(content: dict[str, Any], byte_width: int = 72) -> str:
    buffer = StringIO()
    buffer.write("{\n")
    for k, v in content.items():
        # simple packing of simple values
        if isinstance(v, int | float | bool):
            buffer.write(f"  {k!r}: {v!r},\n")
            continue
        if isinstance(v, str) and len(v) < byte_width:
            buffer.write(f"  {k!r}: {v!r},\n")
            continue
        if (
            isinstance(v, tuple | list)
            and all(isinstance(i, int | float | bool) for i in v)
            and len(repr(v)) < byte_width
        ):
            buffer.write(f"  {k!r}: {v!r},\n")
            continue
        # compact packing of complex values
        compressed_value = serialize_to_bytes(v)
        buffer.write(f"  {k!r}: deserialize_from_bytes(\n")
        while len(compressed_value) > 0:
            buffer.write(f"    {compressed_value[:byte_width]!r}\n")
            compressed_value = compressed_value[byte_width:]
        buffer.write("  ),\n")
    buffer.write("}")
    return buffer.getvalue()
