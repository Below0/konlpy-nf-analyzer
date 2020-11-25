import faust

app = faust.App('nf-worker-1', broker='kafka://49.50.174.75:9092')

class Message(faust.Record):
    date: str
    collected_at: str
    id: str
    ip: str
    title: str
    body: str
    good: int
    bad: int
    is_reply: str

nf_topic = app.topic('naver.finance.board.raw',key_type=str, value_type=Message)

@app.agent(nf_topic, sink=['naver.finance.board'])
async def order(messages):
    async for msg in messages:
        # process infinite stream of orders.
        print(msg)
        yield msg
