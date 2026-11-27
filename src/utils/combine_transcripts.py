import csv
import json
import os

# The video export also lists courses themselves, with this mimeType and resource_id == course_id.
COLLECTION_MIME_TYPE = "application/vnd.ekstep.content-collection"

# The resource list names mime types by their label; the video export uses the mime type itself.
MIME_TYPE_OF_LABEL = {"Video MP4": "video/mp4"}

# Resources (course, mime type and status of each) for the resources that only have a VTT (audio)
# transcript, which the video export does not list. Optional: see combine_transcripts.
RESOURCE_LIST_PATH = "data/input/courses_run_1_vtt.csv"


def read_resource_list(path):
    """resource_id -> its row of the resource list CSV (columns content_id, content_name,
    resource_id, resource_name, mime_type, content_status), mime type labels turned into mime types."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = {}
        for row in csv.DictReader(f):
            row = {key: (value or "").strip() for key, value in row.items()}
            row["mime_type"] = MIME_TYPE_OF_LABEL.get(row.get("mime_type"), row.get("mime_type"))
            rows[row["resource_id"]] = row
        return rows


def combine_transcripts(video_path, language_path, audio_path, output_path, resource_list_path=None):
    """Full outer join of the three transcript files on resource_id.

    A resource in the video export is kept when it is Live there (and is not a course row).
    A resource that is not in the video export (it only has a VTT transcript) is kept when it has
    an audio transcript and the resource list (`resource_list_path`) lists it as Live; its
    course, name, mime type and status then come from the audio file and the resource list. Its
    duration is not known (no file has it).
    """

    def read(path):
        with open(path, encoding="utf-8") as f:
            text = f.read().strip()
        if text.startswith("["):
            return json.loads(text)
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    def blank(v):
        return v is None or (isinstance(v, (str, list, dict)) and len(v) == 0) or (isinstance(v, str) and not v.strip())

    video = {r["identifier"]: r for r in read(video_path)}
    language = {r["resource_id"]: r for r in read(language_path)}  # last one wins on duplicates
    audio = {r["resource_id"]: r for r in read(audio_path)}
    listed = read_resource_list(resource_list_path) if resource_list_path else {}

    def pick(*values):
        return next((v for v in values if not blank(v)), None)

    def pick_course(v, a, r):
        """(course_id, course_name), both from the same file. A resource can be in several
        courses (e.g. a course and a program that reuses its videos) and each file keeps only
        one of them, not always the same one, so taking the id from one file and the name from
        another can pair a course's name with another course's id. The files that have a name
        come first; the video export has none."""
        for course_id, course_name in [
            (a.get("course_id"), a.get("course_name")),
            (r.get("content_id"), r.get("content_name")),
            (v.get("course_id"), None),
        ]:
            if not blank(course_id):
                return course_id, None if blank(course_name) else course_name
        return None, None

    result = []
    vtt_only = 0
    for rid in dict.fromkeys([*video, *language, *audio]):
        v, l, a, r = video.get(rid, {}), language.get(rid, {}), audio.get(rid, {}), listed.get(rid, {})
        if v:
            if v.get("status") != "Live":  # keep only Live resources
                continue
            if v.get("mimeType") == COLLECTION_MIME_TYPE:  # a course row, not a resource
                continue
        else:
            # Only a VTT transcript: kept if the resource list says it is a Live resource.
            if blank(a.get("transcriptions")) or r.get("content_status") != "Live":
                continue
            vtt_only += 1
        course_id, course_name = pick_course(v, a, r)
        result.append({
            "resource_id": rid,
            "name": pick(v.get("name"), a.get("resource_name"), r.get("resource_name")),
            "course_id": course_id,
            "course_name": course_name,
            "status": pick(v.get("status"), r.get("content_status")),
            "mimeType": pick(v.get("mimeType"), r.get("mime_type")),
            "duration_in_sec": pick(v.get("duration_in_sec")),
            "latest_created_timestamp": pick(v.get("latest_created_timestamp")),
            "chunk_count": pick(v.get("chunk_count")),
            "video_transcript": pick(v.get("full_transcript")),
            "source_language": pick(l.get("source_language")),
            "audio_transcript": pick(a.get("transcriptions")),
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Kept {vtt_only} resources that only have a VTT transcript.")
    return len(result)


if __name__ == "__main__":
    d = "data/input/"
    n = combine_transcripts(
        d + "transcripts/video_mp4_transcripts.json",
        d + "transcripts/resource_language_from_audio_transcription.json",
        d + "transcripts/audio_vtt_transcripts_en.json",
        d + "resource_transcripts.json",
        RESOURCE_LIST_PATH if os.path.exists(RESOURCE_LIST_PATH) else None,
    )
    print(f"Wrote {n} records")
