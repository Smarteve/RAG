import string

from hypothesis import given
from hypothesis import strategies as st

from nexusrag.schemas.document import DocumentMetadata

MAX_DOCUMENT_SIZE = 25 * 1024 * 1024

safe_filenames = st.text(
    alphabet=string.ascii_letters + string.digits + "._-",
    min_size=1,
    max_size=251,
)


@given(
    filename=safe_filenames,
    size_bytes=st.integers(min_value=1, max_value=MAX_DOCUMENT_SIZE),
)
def test_valid_metadata_is_accepted_and_filename_is_normalized(
    filename: str,
    size_bytes: int,
) -> None:
    document = DocumentMetadata(
        filename=f"  {filename}  ",
        media_type="application/pdf",
        size_bytes=size_bytes,
    )

    assert document.filename == filename
    assert document.size_bytes == size_bytes
