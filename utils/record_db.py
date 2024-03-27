#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/03/10

from pathlib import Path
import sqlite3

from utils.img_proc import *

DB_FILE = LOG_PATH / 'db.sqlite3'


# https://www.sqlitetutorial.net/
SQL_INIT_DB = '''
CREATE TABLE IF NOT EXISTS Prompt (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  hash TEXT UNIQUE,
  prompt TEXT UNIQUE
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_prompt_hash ON Prompt(hash);
CREATE TABLE IF NOT EXISTS Image (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  hash TEXT UNIQUE,
  filepath TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_image_hash ON Image(hash);
CREATE TABLE IF NOT EXISTS Query (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  pid INTEGER,
  iid INTEGER,
  res TEXT,
  ts_req INTEGER,
  ts_res INTEGER,
  FOREIGN KEY(pid) REFERENCES Prompt(id),
  FOREIGN KEY(iid) REFERENCES Image(id)
);
'''


class RecordDB:

  def __init__(self, fp:Path=DB_FILE) -> None:
    self.cx = sqlite3.connect(fp, isolation_level='IMMEDIATE')
    self.cu = self.cx.cursor()
    self.cu.executescript(SQL_INIT_DB)

  def close(self):
    self.cx.commit()
    self.cx.close()

  def __del__(self):
    self.cx.close()

  def execute_sql(self, sql:str, args:tuple=None):
    if args is not None and not isinstance(args, tuple): args = (args,)
    self.cu.execute(sql, args)
    self.cx.commit()

  def fetch_scalar(self, sql:str, args:tuple=None):
    self.execute_sql(sql, args)
    r = self.cu.fetchone()
    return r[0] if isinstance(r, tuple) else r

  def fetch_results(self, sql:str, args:tuple=None) -> list[tuple]:
    self.execute_sql(sql, args)
    return self.cu.fetchall()

  def get_prompt_id(self, prompt:str) -> int:
    hs = hash_str(prompt)
    r = self.fetch_scalar('SELECT EXISTS(SELECT id FROM Prompt WHERE hash = ?);', hs)
    if r == 0: self.execute_sql('INSERT INTO Prompt(hash, prompt) VALUES (?, ?);', (hs, prompt))
    r = self.fetch_scalar('SELECT id FROM Prompt WHERE hash = ?;', hs)
    return r

  def get_image_id(self, img:PILImage, fp:Path=None) -> int:
    hs = hash_img(img)
    if isinstance(fp, Path): fp = str(fp)
    r = self.fetch_scalar('SELECT EXISTS(SELECT id FROM Image WHERE hash = ?);', hs)
    if r == 0: self.execute_sql('INSERT INTO Image(hash, filepath) VALUES (?, ?);', (hs, fp))
    r = self.fetch_scalar('SELECT id FROM Image WHERE hash = ?;', hs)
    return r

  def add(self, prompt:str, img:PILImage, res:str, ts_req:int=None, ts_res:int=None):
    pid = self.get_prompt_id(prompt)
    iid = self.get_image_id(img)
    if ts_req is not None: ts_req = int(ts_req)
    ts_res = ts_res or now()
    self.execute_sql('INSERT INTO Query(pid, iid, res, ts_req, ts_res) VALUES (?, ?, ?, ?, ?);', (pid, iid, res, ts_req, ts_res))


if __name__ == '__main__':
  db = RecordDB()

  prompt = 'describe the following picture in details'
  pid = db.get_prompt_id(prompt)
  pid2 = db.get_prompt_id(prompt)
  assert pid == pid2

  fp = DATA_RAW_PATH / '0.png'
  img = Image.open(fp)
  iid = db.get_image_id(img, fp)
  iid2 = db.get_image_id(img, fp)
  assert iid == iid2

  db.add(prompt, img, res=r'{"key1": "value1"}')
  db.add(prompt, img, res=r'{"key2": "value2"}', ts_req=123, ts_res=234)
  db.add(prompt, img, res=r'{"key3": "value3"}', ts_res=456)
