from app.schemas import (
    Document,
    KnowledgeBase,
    ChatRequest
)



def test_document():


    doc = Document(

        document_id="doc001",

        filename="G120C.pdf",

        device_model="G120C",

        document_type="manual",

        version="2024",

        file_path="/data/G120C.pdf",

        file_hash="xxxx",

        created_at="2026"

    )


    assert doc.device_model=="G120C"



def test_kb():


    kb = KnowledgeBase(

        knowledge_base_id="kb001",

        name="G120C知识库",

        device_model="G120C",

        document_ids=["doc001"],

        created_at="2026"

    )


    assert kb.document_ids[0]=="doc001"



def test_chat():


    req = ChatRequest(

        knowledge_base_id="kb001",

        device_model="G120C",

        question="F30021怎么处理"

    )


    assert req.top_k==3