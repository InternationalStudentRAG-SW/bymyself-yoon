# Simple RAG Implementation

Due Date: March 22, 2026
Status: In progress

## 🎯 Goal

- [x]  wsl 환경 세팅 (vscode와 연동)
- [x]  python 설치
- [x]  LLM Model(LLAMA Blossom) 설치
- [x]  Langchain 라이브러리 설치
- [x]  테스트 파일 넣어서 테스트 진행

---

## 📑 Ref.

https://jizard.tistory.com/619

https://teddylee777.github.io/langchain/langchain-tutorial-08/

https://m.blog.naver.com/PostView.naver?blogId=se2n&logNo=223443729640&proxyReferer=https:%2F%2Fwww.google.com%2F&trackingCode=external @

---

## 1. 환경 세팅

- 윈도우 가상환경 wsl 사용
    - https://web-yiyeon.tistory.com/12
    - https://cuffyluv.tistory.com/245
- 파이썬
    - https://blog.naver.com/llhwon0103/222504643196
    - https://velog.io/@martinus99/WSL%EC%9D%84-%ED%99%9C%EC%9A%A9%ED%95%B4-Ubuntu-%ED%99%98%EA%B2%BD%EC%97%90%EC%84%9C-Python-%EA%B0%80%EC%83%81-%EA%B0%9C%EB%B0%9C-%ED%99%98%EA%B2%BD-%EA%B5%AC%EC%B6%95%ED%95%98%EA%B8%B0
- llama Bllossom
    - https://seulow-down.tistory.com/357
    - https://hyundoil.tistory.com/434
    - https://god-logger.tistory.com/174#google_vignette ← 이거 따라하기
- langchain
    - `pip install langchain langchain-community`
    - `pip install pypdf chromadb langchain-ollama`
    - 설치 확인용 코드
        
        ```python
        import langchain
        import langchain_community
        import pypdf
        
        print("설치 완료!")
        ```
        
    - 라이브러리 참조 안 될 경우 Ref.
        
        https://www.inflearn.com/community/questions/1722014/3-2-from-langchain-chains-%EC%97%90%EC%84%9C-%EB%AA%A8%EB%93%88%EC%9D%84-%EC%B0%BE%EC%A7%80-%EB%AA%BB%ED%95%A0-%EB%95%8C?srsltid=AfmBOooHznxIMg0kIep7g8M1df73HR1OiMGsVAZGmlXJbrbfsvH5pRfx
        

---

## 2. 모델 테스트

1. 첨부 PDF
    1. https://global.donga.ac.kr/global/CMS/Board/Board.do?mCode=MN066&mode=view&mgr_seq=717&board_seq=8422214
        
        [test_file1.pdf](test_file1.pdf)
        
2. wsl 속 파일 세팅
    
    ![image.png](image.png)
    
3. code
    
    ```python
    from langchain_community.document_loaders import PyPDFLoader # PDF 파일 읽어서 텍스트로 변환 라이브러리
    from langchain_community.vectorstores import Chroma # 텍스트를 숫자 벡터로 저장 후 검색하는 DB
    from langchain_ollama import OllamaEmbeddings # 텍스트를 숫자로 바꾸는 임베딩 작업 수행 Ollama 모델
    from langchain_community.llms import Ollama # Ollama를 통해 LLM과 대화
    from langchain_classic.chains import RetrievalQA # 검색(Retrieval)과 답변 생성(QA) 과정 수행
    
    # 1. PDF 로드
    try:
        # 파일 로드
        loader = PyPDFLoader("/home/suyun/FILE/test_file1.pdf")
    
        # 모델이 읽기 편하도록 페이지 단위나 적절한 크기로 분할
        pages = loader.load_and_split()
    
    except Exception as e:
        print(f"PDF를 읽지 못했습니다. {e}")
    
    # 모델 이름 설정
    B_model = "llama3.2-bllossom-kor-3B:latest"
    
    # 2. 임베딩 및 DB 저장 (Ollama 활용)
    embeddings = OllamaEmbeddings(model=B_model) # e.g., "사과"라는 단어를 [0.1, -0.5, 0.8...]와 같은 좌표값(숫자)로 변환하는 과정
    
    # ChromaDB에 임베딩한 좌표값(숫자) 저장
    docsearch = Chroma.from_documents(pages, embeddings)
    
    # 3. RAG 체인 생성
    llm = Ollama(model=B_model)
    
    # RetrievalQA 체인 생성
    qa = RetrievalQA.from_chain_type(
        llm=llm,                                # 답변을 생성할 LLM 모델
        chain_type="stuff",                     # 검색된 문서 조각들을 모아 모델에게 전달하는 방식 설정
        retriever=docsearch.as_retriever()      # 질문과 관련된 문서를 DB에서 찾아오는 검색 기능 설정
    )
    
    # 4. 질문하기
    while True:
        print("\n질문을 입력해주세요.")
        print("-" * 50)
    
        # 사용자로부터 키보드 입력 수신
        userQ = input(">>> ").strip() # strip(): 앞뒤 공백 제거
    
        # 특정 단어 입력 시 프로그램 종료
        if userQ.lower() in ['exit', '0', '종료']:
            print("프로그램을 종료합니다.")
            break
        
        # 아무것도 입력하지 않은 경우, 응답 대기
        if not userQ:
            continue
    
        print("-" * 50)
        print("답변을 생성하고 있습니다.")
        print("-" * 50)
    
        try:
            # 답변 생성(qa.invoke)
            # qa.invoke 실행 시 수행되는 동작:
                ## (1) 질문 분석 (Embedding)
                ## (2) 관련 내용 찾기 (Retrieval)
                ## (3) 프롬프트 조립 (Augmentation)
                ## (4) 답변 생성 (Generation)
    
            # qa.invoke 실행 시 return값: 딕셔너리 형태
                ## {
                ##      "query": "질문",
                ##      "result": "LLM모델이 생성한 답변 내용",
                ##      "source_documents": [참고한 PDF 조각들] (단, 설정 추가했을 경우에만 출력)
                ## }
            result = qa.invoke(userQ)
    
            # 반환된 딕셔너리 값 중 result만 출력
            print(result['result'])
            print("-" * 50)
            
        except Exception as e:
            print(f"\n답변 생성 중 오류가 발생했습니다. {e}")
    ```
    
4. result
    
    ![image.png](image%201.png)
    
    [https://www.notion.so](https://www.notion.so)
    
5. 개선해야 할 점
    1. 중간중간에 한글 출력이 제대로 되지 않는 부분이 있음
        
        ![숫자 출력 오류](image%202.png)
        
        숫자 출력 오류
        
        ![제출해야 할 서류](image%203.png)
        
        제출해야 할 서류
        
    2. 가독성 높이면 좋을 것 같음
6. 앞으로 시도 및 알아볼 것
    1. PDF 여러개 넣고 출력해보기
    2. ChormaDB 관리 어떻게 해야하는지 (상관없나?)
