from fastapi import FastAPI


app = FastAPI(debug=True)


@app.post("/v1/availability-test")
async def availability_test() -> int:
    return 200
