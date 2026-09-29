"""
증권사 연결 인터페이스. NH선물 REST API(조회·주문) + 웹소켓(실시간 시세) 구현은 **공식 문서를 받은 뒤** 작성한다.
엔드포인트·파라미터·종목코드 형식은 추측하지 말 것 (CLAUDE.md 10번). 기본값은 항상 모의(paper=True).
키는 환경변수로만 (예: NH_APP_KEY, NH_APP_SECRET — 이름은 발급 후 확정).
NH 한도 (2026-09-29 답변): 웹소켓 구독 20종목, 주문 초당 5건, 시세 초당 10건, 계좌 초당 10건. SPXW·XSP 모두 가능, 모의 서버도 CBOE 실시간.
"""
from abc import ABC, abstractmethod
import pandas as pd


class Broker(ABC):
    paper: bool = True

    @abstractmethod
    def mnq_1m_bars(self, since: pd.Timestamp) -> pd.DataFrame:
        """MNQ 근월물 1분봉 [open, high, low, close], 인덱스 = 봉 시작 (뉴욕 시간). 전날 저녁부터 필요."""

    @abstractmethod
    def spx_0dte_chain(self) -> pd.DataFrame:
        """오늘 만기 SPXW 호가 [strike, call_bid, call_ask, put_bid, put_ask]. 웹소켓 20종목 한도 → ATM ±4행사가(콜·풋 16개)만, 중심은 09:31 전후 다시 잡기."""

    @abstractmethod
    def place_limit_buy(self, strike: float, right: str, qty: int, price: float) -> str:
        """지정가 매수 주문 → 주문번호. right = 'C'."""

    @abstractmethod
    def order_status(self, order_id: str) -> dict:
        """{'filled': bool, 'fill_price': float|None}"""

    @abstractmethod
    def cancel(self, order_id: str) -> None: ...


class NHFuturesBroker(Broker):
    """NH선물 REST API — 문서 받으면 채움. 지금은 일부러 비워 둠."""

    def __init__(self, paper=True):
        self.paper = paper
        raise NotImplementedError("NH선물 REST API 공식 문서를 받은 뒤 구현 (모의 서버부터)")

    def mnq_1m_bars(self, since): ...
    def spx_0dte_chain(self): ...
    def place_limit_buy(self, strike, right, qty, price): ...
    def order_status(self, order_id): ...
    def cancel(self, order_id): ...
