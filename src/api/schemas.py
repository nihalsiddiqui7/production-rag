from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    question: str = Field(
        min_length=3, 
        max_length=1000, 
        description="User Question"
    )

class SourceInfo(BaseModel):
    page: int = Field(
        description="Page number of the source document"
    )
    title: str = Field(
        description="Title of the source document"
    )


class QueryResponse(BaseModel):
    answer: str = Field(
        description="Answer to the user's question"
    )
    sources: list[SourceInfo]



