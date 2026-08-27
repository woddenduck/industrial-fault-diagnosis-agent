import os
import json
import hashlib
import uuid
import shutil

from app.config import DATA_DIR, UPLOAD_DIR


UPLOAD_DIR = str(UPLOAD_DIR)
DB_FILE = str(DATA_DIR / "documents.json")


ALLOWED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".md"
}



def calculate_sha256(file_path):

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:

        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)

    return sha256.hexdigest()



def load_documents():

    if not os.path.exists(DB_FILE):

        return []

    with open(
        DB_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)



def save_documents(documents):

    os.makedirs(
        os.path.dirname(DB_FILE),
        exist_ok=True
    )

    with open(
        DB_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            documents,
            f,
            ensure_ascii=False,
            indent=4
        )



def validate_file(file_path):

    if not os.path.exists(file_path):

        raise ValueError(
            "文件不存在"
        )


    if os.path.getsize(file_path)==0:

        raise ValueError(
            "空文件"
        )


    ext = os.path.splitext(
        file_path
    )[1].lower()


    if ext not in ALLOWED_EXTENSIONS:

        raise ValueError(
            f"不支持文件类型:{ext}"
        )



def save_document(
        file_path,
        device_model,
        document_type,
        version=None
):

    # 参数检查

    if not device_model:

        raise ValueError(
            "device_model不能为空"
        )


    if not document_type:

        raise ValueError(
            "document_type不能为空"
        )


    # 文件检查

    validate_file(file_path)



    # hash

    file_hash = calculate_sha256(
        file_path
    )


    documents = load_documents()



    # duplicate检测

    for doc in documents:

        if doc["sha256"] == file_hash:

            return {
                "status":"duplicate",
                "document_id":
                    doc["document_id"]
            }



    # 创建ID

    document_id = (
        "doc_"
        +
        uuid.uuid4()
        .hex[:8]
    )



    save_dir = os.path.join(
        UPLOAD_DIR,
        document_id
    )


    os.makedirs(
        save_dir,
        exist_ok=True
    )


    filename=os.path.basename(
        file_path
    )


    target=os.path.join(
        save_dir,
        filename
    )


    shutil.copy2(
        file_path,
        target
    )



    metadata={

        "document_id":
            document_id,

        "filename":
            filename,

        "sha256":
            file_hash,

        "device_model":
            device_model,

        "document_type":
            document_type,

        "version":
            version,

        "path":
            target
    }



    documents.append(
        metadata
    )


    save_documents(
        documents
    )


    return {

        "status":"success",

        "document_id":
            document_id
    }
