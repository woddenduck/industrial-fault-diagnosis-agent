from app.services.document_service import save_document



def test_success_pdf():

    result = save_document(

        "test_files/test.pdf",

        "G120C",

        "manual",

        "v1"

    )

    print(result)



def test_success_txt():

    result = save_document(

        "test_files/test.txt",

        "G120C",

        "note",

        "v1"

    )

    print(result)



def test_success_md():

    result = save_document(

        "test_files/test.md",

        "G120C",

        "manual",

        "v1"

    )

    print(result)



def test_duplicate():

    result = save_document(

        "test_files/test.pdf",

        "G120C",

        "manual",

        "v1"

    )

    print(result)



def test_jpg():

    try:

        save_document(

            "test_files/test.jpg",

            "G120C",

            "manual"

        )

    except Exception as e:

        print(
            "拒绝:",
            e
        )



def test_empty():

    try:

        save_document(

            "test_files/empty.txt",

            "G120C",

            "manual"

        )

    except Exception as e:

        print(
            "拒绝:",
            e
        )



if __name__=="__main__":


    test_success_pdf()

    test_success_txt()

    test_success_md()

    test_duplicate()

    test_jpg()

    test_empty()