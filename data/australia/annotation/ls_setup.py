"""Create the LS project + import tiles/proposals via the local API.

Run AFTER you've signed up in the browser (so a user exists). Reads/mints the
user's API token straight from the LS sqlite DB (never printed), creates the
project with the protocol config, and imports the tasks (with editable polygon
pre-annotations). Local-files serving is already enabled, so no storage step.
"""
import json, os, secrets, sqlite3, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.expanduser("~/Library/Application Support/label-studio/label_studio.sqlite3")
BASE = "http://localhost:8080"
TASKS = os.path.join(HERE, "exports", "ls_tasks_won_try.json")
CONFIG = open(os.path.join(HERE, "exports", "labeling_config.xml")).read()
TITLE = "Dryland ITC — WON try"


def get_token():
    con = sqlite3.connect(DB); cur = con.cursor()
    u = cur.execute("SELECT id,email FROM htx_user ORDER BY id LIMIT 1").fetchone()
    if not u:
        raise SystemExit("No Label Studio user yet — sign up in the browser first.")
    uid, email = u
    row = cur.execute("SELECT key FROM authtoken_token WHERE user_id=?", (uid,)).fetchone()
    if row:
        key = row[0]
    else:
        key = secrets.token_hex(20)
        cur.execute("INSERT INTO authtoken_token(key,created,user_id) VALUES(?,?,?)",
                    (key, time.strftime("%Y-%m-%d %H:%M:%S"), uid))
        con.commit()
    con.close(); return key, email


def api(method, path, tok, data=None):
    req = urllib.request.Request(
        BASE + path, method=method,
        headers={"Authorization": f"Token {tok}", "Content-Type": "application/json"},
        data=json.dumps(data).encode() if data is not None else None)
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read() or "{}")


def main():
    tok, email = get_token()
    print("user:", email)
    proj = api("POST", "/api/projects", tok, {"title": TITLE, "label_config": CONFIG})
    pid = proj["id"]
    # Register a local storage so LS is allowed to SERVE the tiles (do NOT sync it,
    # or it would auto-create duplicate tasks from the folder).
    api("POST", "/api/storages/localfiles", tok, {
        "project": pid, "path": os.path.join(HERE, "tiles"),
        "regex_filter": r".*\.png", "use_blob_urls": True, "title": "tiles"})
    tasks = json.load(open(TASKS))
    res = api("POST", f"/api/projects/{pid}/import", tok, tasks)
    print(f"project {pid}: imported {res.get('task_count', res)} tasks, "
          f"{res.get('prediction_count','?')} predictions")
    print(f"OPEN ON IPAD: http://100.124.137.30:8080/projects/{pid}/data")


if __name__ == "__main__":
    main()
