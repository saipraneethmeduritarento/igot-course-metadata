import json

from src.utils.combine_transcripts import COLLECTION_MIME_TYPE, combine_transcripts

RESOURCE_LIST = (
    "content_id,content_name,resource_id,resource_name,resource_type,mime_type,content_status\n"
    "c1,Course One,vtt1,VTT One,Learning Resource,Video MP4,Live\n"
    "c1,Course One,vtt_empty,VTT Empty,Learning Resource,Video MP4,Live\n"
    "c1,Course One,vtt_retired,VTT Retired,Learning Resource,Video MP4,Retired\n"
)


def write_inputs(tmp_path, resource_list=True):
    video = [
        {"identifier": "v1", "status": "Live", "mimeType": "video/mp4", "course_id": "c0",
         "name": "Video One", "duration_in_sec": "60", "full_transcript": "video text"},
        {"identifier": "v_draft", "status": "Draft", "mimeType": "video/mp4", "full_transcript": "x"},
        {"identifier": "c0", "status": "Live", "mimeType": COLLECTION_MIME_TYPE},
    ]
    language = [{"resource_id": "vtt1", "source_language": ["English"]}]
    audio = [
        {"resource_id": "vtt1", "resource_name": "VTT One", "course_id": "c1",
         "course_name": "Course One", "transcriptions": "audio text"},
        {"resource_id": "vtt_empty", "course_id": "c1", "transcriptions": ""},
        {"resource_id": "vtt_retired", "course_id": "c1", "transcriptions": "old"},
        {"resource_id": "vtt_unlisted", "course_id": "c2", "transcriptions": "text"},
    ]
    paths = {}
    for name, records in [("video", video), ("language", language), ("audio", audio)]:
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(records))
    paths["list"] = tmp_path / "resources.csv"
    paths["list"].write_text(RESOURCE_LIST)
    paths["out"] = tmp_path / "out.json"
    return paths


def combine(paths, with_list=True):
    combine_transcripts(
        paths["video"], paths["language"], paths["audio"], paths["out"],
        paths["list"] if with_list else None,
    )
    return {r["resource_id"]: r for r in json.loads(paths["out"].read_text())}


def test_live_video_resources_are_kept_and_others_dropped(tmp_path):
    records = combine(write_inputs(tmp_path))
    assert "v1" in records
    assert "v_draft" not in records  # not Live
    assert "c0" not in records  # a course row


def test_vtt_only_resource_listed_as_live_is_kept_with_its_list_details(tmp_path):
    record = combine(write_inputs(tmp_path))["vtt1"]
    assert record["course_id"] == "c1"
    assert record["mimeType"] == "video/mp4"  # from the label "Video MP4"
    assert record["status"] == "Live"
    assert record["audio_transcript"] == "audio text"
    assert record["source_language"] == ["English"]
    assert record["video_transcript"] is None
    assert record["duration_in_sec"] is None


def test_vtt_only_resource_without_text_not_live_or_not_listed_is_dropped(tmp_path):
    records = combine(write_inputs(tmp_path))
    assert "vtt_empty" not in records
    assert "vtt_retired" not in records
    assert "vtt_unlisted" not in records


def test_without_a_resource_list_only_video_resources_are_kept(tmp_path):
    assert set(combine(write_inputs(tmp_path), with_list=False)) == {"v1"}


def test_course_id_and_name_come_from_the_same_file(tmp_path):
    # The resource is in a course and in a program; the video export names the program, the
    # audio export the course. The id must not come from one and the name from the other.
    paths = write_inputs(tmp_path)
    video = json.loads(paths["video"].read_text())
    video.append({"identifier": "shared", "status": "Live", "mimeType": "video/mp4",
                  "course_id": "program", "full_transcript": "video text"})
    paths["video"].write_text(json.dumps(video))
    audio = json.loads(paths["audio"].read_text())
    audio.append({"resource_id": "shared", "course_id": "course", "course_name": "The Course",
                  "transcriptions": "audio text"})
    paths["audio"].write_text(json.dumps(audio))

    record = combine(paths)["shared"]
    assert (record["course_id"], record["course_name"]) == ("course", "The Course")


def test_video_only_resource_keeps_its_course_id_without_a_name(tmp_path):
    record = combine(write_inputs(tmp_path))["v1"]
    assert (record["course_id"], record["course_name"]) == ("c0", None)
