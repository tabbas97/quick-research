import arxiv
from typing import List
import re
from typing import Union

def valid_arxiv_id(paper_id : str):
    if re.search(r"\d{4}\.\d{5}",paper_id) is None:
        return False
    return True

def retrieve_papers(paper_id : Union[str,List[str]]):
    
    if isinstance(paper_id,str):
        paper_id = [paper_id]
            
    for paper in paper_id:
        if not valid_arxiv_id(paper):
            raise ValueError(f"Invalid arxiv id {paper}")
        
    client = arxiv.Client()
    search = arxiv.Search(id_list=paper_id)
    results = client.results(search)
    
    allRes = []
    
    for res in results:
        res.download_pdf("pdfs",f"{res.get_short_id()}.pdf")
        allRes.append(res)
    
    return allRes

if __name__ == "__main__":
    
    papers : List[arxiv.Result] = retrieve_papers("2403.14589")
    print(papers)


