from fastapi import FastAPI,Body
import uvicorn
from pydantic import BaseModel,Field,model_validator,field_validator
from typing import Literal,Annotated
from generate_poem import generate_poem
import re

class RequestBody(BaseModel):
    starter:str=Field(...,max_length=300)
    rhyme_scheme:Annotated[str,Field(...)]
    mode:Literal["line_based","poem_based"]=Field(default="line_based")
    @model_validator(mode="after")
    @classmethod
    def check_rhyme_scheme_length(cls, values):
        mode = values.mode
        rhyme_scheme = values.rhyme_scheme   
        if mode == "poem_based" and len(rhyme_scheme) >= 5:
            raise ValueError("Şiir bazlı mod sadece kısa şiirler içindir. Lütfen kafiye şemasını bir dörtlüğe indirgeyiniz.")
        return values
    @field_validator("rhyme_scheme",mode="after")
    @classmethod
    def check_rhyme_scheme(cls,rhyme_scheme,values):
        if re.match(r'^[A-Z]+$',rhyme_scheme) is None:
            raise ValueError("Kafiye şeması büyük harflerden oluşmalıdır")
        return rhyme_scheme
    
app=FastAPI()

@app.post(path="/generate")
def generate_poem_api(body:Annotated[RequestBody,Body(strict=True)]):
    poem_info={
        "starter":body.starter,
        "rhyme_scheme":body.rhyme_scheme,
        "mode":body.mode,
    }
    poem=generate_poem(**poem_info)
    return {
        "poem":poem
    }
if __name__=="__main__":
    uvicorn.run(app,port=8080)