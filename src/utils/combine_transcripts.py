import json


def combine_transcripts(video_path, language_path, audio_path, output_path):
    """Full outer join of the three transcript files on resource_id."""

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

    def pick(*values):
        return next((v for v in values if not blank(v)), None)

    result = []
    for rid in dict.fromkeys([*video, *language, *audio]):
        v, l, a = video.get(rid, {}), language.get(rid, {}), audio.get(rid, {})
        if v.get("status") != "Live":  # keep only Live resources
            continue
        result.append({
            "resource_id": rid,
            "name": pick(v.get("name"), a.get("resource_name")),
            "course_id": pick(v.get("course_id"), a.get("course_id")),
            "course_name": pick(a.get("course_name")),
            "status": pick(v.get("status")),
            "mimeType": pick(v.get("mimeType")),
            "duration_in_sec": pick(v.get("duration_in_sec")),
            "latest_created_timestamp": pick(v.get("latest_created_timestamp")),
            "chunk_count": pick(v.get("chunk_count")),
            "video_transcript": pick(v.get("full_transcript")),
            "source_language": pick(l.get("source_language")),
            "audio_transcript": pick(a.get("transcriptions")),
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return len(result)


if __name__ == "__main__":
    d = "data/input/"
    n = combine_transcripts(
        d + "transcripts/video_mp4_transcripts.json",
        d + "transcripts/resource_language_from_audio_transcription.json",
        d + "transcripts/audio_vtt_transcripts_en.json",
        d + "resource_transcripts.json",
    )
    print(f"Wrote {n} records")
