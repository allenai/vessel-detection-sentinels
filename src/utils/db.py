def dict_factory(cursor, row):
    """Converts rows returned from sqlite queries to dicts."""
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d


def get_dataset(conn, id):
    query = f"SELECT d.id, d.collection_id, d.name, d.task, d.categories FROM datasets AS d WHERE d.id = {id}"
    cur = conn.cursor()
    row = cur.execute(query).fetchone()
    return row


def get_windows(conn, dataset_id, split=None):
    query = f"SELECT w.id, w.dataset_id, w.image_id, w.row, w.column, w.height, w.width, w.hidden, w.split FROM windows AS w WHERE w.dataset_id = {dataset_id}"
    if split:
        query += f' AND w.split = "{split}"'
    cur = conn.cursor()
    rows = cur.execute(query).fetchall()
    return rows


def get_image(conn, image_id):
    query = f"SELECT ims.id, ims.uuid, ims.name, ims.format, ims.channels, ims.width, ims.height, ims.preprocessed, ims.hidden, ims.bounds, ims.time, ims.projection, ims.column, ims.row, ims.zoom FROM images AS ims WHERE ims.id = {image_id}"
    cur = conn.cursor()
    row = cur.execute(query).fetchone()
    return row


def get_labels(conn, window_id):
    query = f"SELECT l.id, l.window_id, l.row, l.column, l.height, l.width, l.extent, l.value, l.properties FROM labels AS l WHERE l.window_id = {window_id}"
    cur = conn.cursor()
    rows = cur.execute(query).fetchall()
    return rows


def get_dataset_labels(conn, dataset_id, splits=[]):
    split_params = ""
    if len(splits) > 0:
        split_params += " AND"
        for idx, split in enumerate(splits):
            split_params += f" split='{split}'"
            if idx < (len(splits) - 1):
                split_params += " OR"

    query = f"SELECT l.id, l.window_id, l.row, l.column, l.height, l.width, l.extent, l.value, l.properties FROM labels AS l WHERE l.window_id in (SELECT id from windows where dataset_id={dataset_id}{split_params})"
    cur = conn.cursor()
    rows = cur.execute(query).fetchall()
    return rows
