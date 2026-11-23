// Stage 4 Read-Only SQLite Translation Store (`culinary_menu.db`) using Apple's built-in libsqlite3.
// Zero third-party dependencies; opens strictly SQLITE_OPEN_READONLY.

import Foundation
import SQLite3

final class Stage4SQLiteStore {
    private var db: OpaquePointer?
    private let lock = NSLock()

    func open(path: String) throws {
        lock.lock()
        defer { lock.unlock() }
        if db != nil { return }
        var handle: OpaquePointer?
        let rc = sqlite3_open_v2(path, &handle, SQLITE_OPEN_READONLY, nil)
        guard rc == SQLITE_OK, let opened = handle else {
            let msg = handle.flatMap { String(cString: sqlite3_errmsg($0)) } ?? "sqlite3_open_v2 failed (\(rc))"
            if let h = handle { sqlite3_close(h) }
            throw NSError(domain: "Stage4SQLiteStore", code: Int(rc), userInfo: [NSLocalizedDescriptionKey: msg])
        }
        db = opened
    }

    deinit {
        if let handle = db {
            sqlite3_close(handle)
        }
    }

    func lookupTerms(_ terms: [String]) -> [String: [String: Any]] {
        lock.lock()
        defer { lock.unlock() }
        guard let db = db, !terms.isEmpty else { return [:] }

        var out: [String: [String: Any]] = [:]
        let sql = """
            SELECT m.id, m.canonical_jp, m.reading_hiragana,
                   COALESCE(NULLIF(o.curated_name_en, ''), m.name_en) AS effective_en
            FROM menu_dictionary m
            LEFT JOIN curated_menu_overrides o ON o.menu_id = m.id
            WHERE m.canonical_jp = ?1
            LIMIT 1
            """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK, let s = stmt else {
            return [:]
        }
        defer { sqlite3_finalize(s) }

        let transient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
        for term in terms {
            sqlite3_reset(s)
            sqlite3_clear_bindings(s)
            sqlite3_bind_text(s, 1, (term as NSString).utf8String, -1, transient)
            if sqlite3_step(s) == SQLITE_ROW {
                let dbId = Int(sqlite3_column_int64(s, 0))
                let jp = sqlite3_column_text(s, 1).map { String(cString: $0) } ?? term
                let reading = sqlite3_column_text(s, 2).map { String(cString: $0) } ?? ""
                let en = sqlite3_column_text(s, 3).map { String(cString: $0) } ?? ""
                out[term] = [
                    "db_id": dbId,
                    "canonical_jp": jp,
                    "reading_hiragana": reading,
                    "name_en": en,
                ]
            }
        }
        return out
    }

    func lookupPhrasePlan(_ canonicalPhrase: String) -> [String: Any]? {
        lock.lock()
        defer { lock.unlock() }
        guard let db = db, !canonicalPhrase.isEmpty else { return nil }

        let sql = """
            SELECT canonical_phrase, phrase_en, provenance_source
            FROM curated_phrase_components
            WHERE canonical_phrase = ?1
            LIMIT 1
            """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK, let s = stmt else {
            return nil
        }
        defer { sqlite3_finalize(s) }

        let transient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
        sqlite3_bind_text(s, 1, (canonicalPhrase as NSString).utf8String, -1, transient)
        if sqlite3_step(s) == SQLITE_ROW {
            let phrase = sqlite3_column_text(s, 0).map { String(cString: $0) } ?? canonicalPhrase
            let en = sqlite3_column_text(s, 1).map { String(cString: $0) } ?? ""
            let prov = sqlite3_column_text(s, 2).map { String(cString: $0) } ?? ""
            return [
                "canonical_phrase": phrase,
                "phrase_en": en,
                "provenance_source": prov,
            ]
        }
        return nil
    }
}
