# -*- coding: utf-8 -*-
# @Time    : 2022/2/17 16:55
# @Author  : Jiangzhesheng
# @File    : number.py
# @Software: PyCharm
# @Description:判断有效数字，用于判断一个数据是一个离散型字符串还是一个连续性数值
from enum import Enum
def isNumber(s: str) -> bool:
    State = Enum("State", [
        "STATE_INITIAL",
        "STATE_INT_SIGN",
        "STATE_INTEGER",
        "STATE_POINT",
        "STATE_POINT_WITHOUT_INT",
        "STATE_FRACTION",
        "STATE_EXP",
        "STATE_EXP_SIGN",
        "STATE_EXP_NUMBER",
        "STATE_END"
    ])
    Chartype = Enum("Chartype", [
        "CHAR_NUMBER",
        "CHAR_EXP",
        "CHAR_POINT",
        "CHAR_SIGN",
        "CHAR_ILLEGAL"
    ])

    def toChartype(ch: str) -> Chartype:
        if ch.isdigit():
            return Chartype.CHAR_NUMBER
        elif ch.lower() == "e":
            return Chartype.CHAR_EXP
        elif ch == ".":
            return Chartype.CHAR_POINT
        elif ch == "+" or ch == "-":
            return Chartype.CHAR_SIGN
        else:
            return Chartype.CHAR_ILLEGAL

    transfer = {
        State.STATE_INITIAL: {
            Chartype.CHAR_NUMBER: State.STATE_INTEGER,
            Chartype.CHAR_POINT: State.STATE_POINT_WITHOUT_INT,
            Chartype.CHAR_SIGN: State.STATE_INT_SIGN
        },
        State.STATE_INT_SIGN: {
            Chartype.CHAR_NUMBER: State.STATE_INTEGER,
            Chartype.CHAR_POINT: State.STATE_POINT_WITHOUT_INT
        },
        State.STATE_INTEGER: {
            Chartype.CHAR_NUMBER: State.STATE_INTEGER,
            Chartype.CHAR_EXP: State.STATE_EXP,
            Chartype.CHAR_POINT: State.STATE_POINT
        },
        State.STATE_POINT: {
            Chartype.CHAR_NUMBER: State.STATE_FRACTION,
            Chartype.CHAR_EXP: State.STATE_EXP
        },
        State.STATE_POINT_WITHOUT_INT: {
            Chartype.CHAR_NUMBER: State.STATE_FRACTION
        },
        State.STATE_FRACTION: {
            Chartype.CHAR_NUMBER: State.STATE_FRACTION,
            Chartype.CHAR_EXP: State.STATE_EXP
        },
        State.STATE_EXP: {
            Chartype.CHAR_NUMBER: State.STATE_EXP_NUMBER,
            Chartype.CHAR_SIGN: State.STATE_EXP_SIGN
        },
        State.STATE_EXP_SIGN: {
            Chartype.CHAR_NUMBER: State.STATE_EXP_NUMBER
        },
        State.STATE_EXP_NUMBER: {
            Chartype.CHAR_NUMBER: State.STATE_EXP_NUMBER
        },
    }

    st = State.STATE_INITIAL
    for ch in s:
        typ = toChartype(ch)
        if typ not in transfer[st]:
            return False
        st = transfer[st][typ]

    return st in [State.STATE_INTEGER, State.STATE_POINT, State.STATE_FRACTION, State.STATE_EXP_NUMBER,
                  State.STATE_END]