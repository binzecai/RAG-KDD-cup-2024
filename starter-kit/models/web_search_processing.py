from blingfire import text_to_sentences_and_offsets
from bs4 import BeautifulSoup
from langchain.schema import Document

def extract_web_sentences(search_results, max_len=256):
    all_chunks = []
    for idx,html_text in enumerate(search_results):
        chunks = []
        soup = BeautifulSoup(
            html_text["page_result"], features="html.parser"
        )
        text = soup.get_text().replace("\n", "")
        if len(text) > 0:
            offsets = text_to_sentences_and_offsets(text)[1]
            current_chunk = ""
            for ofs in offsets:
                # Extract each sentence based on its offset and limit its length.
                sentence = text[ofs[0] : ofs[1]]
                if len(current_chunk) + len(sentence) <= max_len:
                    current_chunk += sentence
                elif len(current_chunk) > 0:
                    chunks.append(Document(page_content=current_chunk, metadata={'web':idx+1}))
                    current_chunk = ""
                if current_chunk:
                    chunks.append(Document(page_content=current_chunk, metadata={'web':idx+1}))
        # merge context
        cleaned_chunks = [] 
        i = 0
        while i <= len(chunks)-2:
            current_chunk = chunks[i]
            next_chunk = chunks[min(i+1, len(chunks)-1)] 
            if len(next_chunk.page_content) < 0.5 * len(current_chunk.page_content):
                new_chunk = Document(page_content=current_chunk.page_content + next_chunk.page_content, metadata=current_chunk.metadata)
                cleaned_chunks.append(new_chunk)
                i += 2
            else:
                i+=1
                cleaned_chunks.append(current_chunk)
        all_chunks += chunks
    return all_chunks