"""Read-only, bounded AIS endpoints for the Traffic Map page."""

from flask import Blueprint, abort, current_app, jsonify, request
from werkzeug.exceptions import HTTPException

from data_io.ais import load_ais, parse_utc


traffic_bp = Blueprint("traffic_data", __name__, url_prefix="/api/traffic")


@traffic_bp.errorhandler(HTTPException)
def http_error(error):
    return jsonify(error=error.description), error.code


def store():
    try:
        return load_ais(current_app.config["AIS_CSV"])
    except (OSError, ValueError) as error:
        abort(503, description=f"AIS 데이터를 읽을 수 없습니다: {error}")


@traffic_bp.get("/meta")
def metadata():
    return jsonify(store().metadata())


@traffic_bp.get("/positions")
def positions():
    data = store()
    at_text = request.args.get("at", data.start_utc)
    try:
        at = parse_utc(at_text)
        label = request.args.get("class", "").strip()
        vessels = data.positions_at(at, label)
    except ValueError as error:
        abort(400, description=str(error))
    return jsonify(at_utc=at.isoformat(timespec="seconds").replace("+00:00", "Z"),
                   count=len(vessels), vessels=vessels)


@traffic_bp.get("/track/<mmsi>")
def track(mmsi):
    if len(mmsi) != 9 or not mmsi.isdigit():
        abort(400, description="MMSI는 9자리 숫자여야 합니다.")
    try:
        result = store().vessel_track(mmsi)
    except KeyError:
        abort(404, description="해당 선박의 항적이 없습니다.")
    return jsonify(result)
