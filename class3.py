import pymysql
import pandas as pd
import numpy as np 

class Connection:
    def __init__(self, host, id, pw, db_name):
        self.con = pymysql.connect(
            host=host, 
            user= id, 
            password=pw, 
            db=db_name, 
            charset='utf8',
            cursorclass = pymysql.cursors.DictCursor
            )        
        self.cur = self.con.cursor()

    def _select(self,sql,args=None):
        self.cur.execute(sql,args)
        self.sel = self.cur.fetchone()
        self.cur.close()
        self.con.close()
        return self.sel

    def _selectAll(self,sql,args=None):
        self.cur.execute(sql,args)
        self.sel = self.cur.fetchall()
        self.cur.close()
        self.con.close()
        return self.sel

    def _insert(self,sql,args=None):
        self.ins = self.cur.executemany(sql,args)
        return self.ins

    def _update(self,sql, args=None):
        self.upd = self.cur.executemany(sql,args)
        return self.upd

    def _delete(self, sql, args=None):
        self.delete = self.cur.executemany(sql,args)
        return self.delete
    