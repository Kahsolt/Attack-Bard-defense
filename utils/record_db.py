#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/03/10

from pathlib import Path
import sqlite3

from utils.img_proc import *

DB_FILE = LOG_PATH / 'record.db'


# https://www.sqlitetutorial.net/
SQL_INIT_DB = '''
-- Enity tables: table_name and field_name must be accordingly to
CREATE TABLE IF NOT EXISTS Prompt (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  hash TEXT UNIQUE,
  prompt TEXT UNIQUE
);
CREATE TABLE IF NOT EXISTS Image (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  hash TEXT UNIQUE,
  image BLOB UNIQUE
);
-- Relation tables
CREATE TABLE IF NOT EXISTS Query (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  pid INTEGER,
  iid INTEGER,
  res TEXT,
  provider TEXT,
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

  def get_entity_id(self, table_name:str, hash:str, data:Any=None) -> int:
    field_name = table_name.lower()
    r = self.fetch_scalar(f'SELECT EXISTS(SELECT id FROM {table_name} WHERE hash = ?);', hash)
    if r == 0: self.execute_sql(f'INSERT INTO {table_name}(hash, {field_name}) VALUES (?, ?);', (hash, data))
    r = self.fetch_scalar(f'SELECT id FROM {table_name} WHERE hash = ?;', hash)
    return r

  def get_prompt_id(self, key:Union[int, str]) -> int:
    if isinstance(key, int): return key
    prompt = key
    assert isinstance(prompt, str)
    hash = hash_bdata(prompt.encode(), method='md5')  # len=128
    return self.get_entity_id('Prompt', hash, key)

  def get_image_id(self, key:Union[int, Path, bytes]) -> int:
    if isinstance(key, int): return key
    bdata = read_file(key) if isinstance(key, Path) else key
    assert isinstance(bdata, bytes)
    hash = hash_bdata(bdata, method='sha512')        # len=512
    return self.get_entity_id('Image', hash, None)

  def has(self, prompt:Union[int, str], image:Union[int, Path, bytes]) -> bool:
    pid = self.get_prompt_id(prompt)
    iid = self.get_image_id(image)
    r = self.fetch_scalar('SELECT EXISTS(SELECT id FROM Query WHERE pid = ? AND iid = ?);', (pid, iid))
    return bool(r)

  def add(self, prompt:Union[int, str], image:Union[int, Path, bytes], res:str, ts_req:int=None, ts_res:int=None, provider:str=None):
    pid = self.get_prompt_id(prompt)
    iid = self.get_image_id(image)
    ts_req = ts_req and int(ts_req)
    ts_res = ts_res or now()
    self.execute_sql('INSERT INTO Query(pid, iid, res, provider, ts_req, ts_res) VALUES (?, ?, ?, ?, ?, ?);', (pid, iid, res, provider, ts_req, ts_res))


if __name__ == '__main__':
  db = RecordDB(LOG_PATH / 'test.db')

  prompt = 'describe the following picture in details'
  pid  = db.get_prompt_id(prompt)
  pid2 = db.get_prompt_id(prompt)
  assert pid == pid2

  fp = DATA_RAW_PATH / '0.png'
  iid  = db.get_image_id(fp)
  iid2 = db.get_image_id(fp)
  assert iid == iid2

  db.add(pid, iid, res='{"key1": "value1"}')
  db.add(pid, iid, res='{"key2": "value2"}', ts_req=123, ts_res=234, provider='local')
  db.add(pid, iid, res='{"key3": "value3"}', ts_res=456)
