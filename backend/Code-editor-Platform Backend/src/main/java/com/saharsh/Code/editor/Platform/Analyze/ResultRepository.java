package com.saharsh.Code.editor.Platform.Analyze;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Repository;

import java.sql.ResultSet;
import java.util.Base64;
import java.util.List;

@Repository
public class ResultRepository {

    private final JdbcTemplate jdbc;
    private final ObjectMapper mapper;

    public ResultRepository(JdbcTemplate jdbc, ObjectMapper mapper) {
        this.jdbc = jdbc;
        this.mapper = mapper;
    }

    public List<Row> fetchPhoneRows(long userId, long testId) {
        String sql = """
                SELECT user_id, test_id, time, is_phone, processed_at, result, image
                FROM phone_result
                WHERE user_id = ? AND test_id = ?
                """;
        return jdbc.query(sql, (rs, i) -> mapRow(rs), userId, testId);
    }

    public List<Row> fetchLaptopRows(long userId, long testId) {
        String sql = """
                SELECT user_id, test_id, time, is_phone, processed_at, result, image
                FROM laptop_process_result
                WHERE user_id = ? AND test_id = ?
                """;
        return jdbc.query(sql, (rs, i) -> mapRow(rs), userId, testId);
    }

    private Row mapRow(ResultSet rs) throws java.sql.SQLException {
        Row r = new Row();
        r.userId = rs.getLong("user_id");
        r.testId = rs.getLong("test_id");
        r.time = rs.getLong("time");
        r.isPhone = rs.getBoolean("is_phone");
        r.processedAt = rs.getTimestamp("processed_at");

        // JSONB -> JsonNode
        String json = rs.getString("result");
        try { r.result = mapper.readTree(json); }
        catch (Exception e) { r.result = mapper.nullNode(); }

        // BYTEA / BLOB -> Base64
        byte[] bytes = rs.getBytes("image");            // null-safe; returns null if column is NULL
        r.base64Image = (bytes == null) ? null : Base64.getEncoder().encodeToString(bytes);

        return r;
    }

    public static class Row {
        public long userId;
        public long testId;
        public long time;
        public boolean isPhone;
        public java.util.Date processedAt;
        public JsonNode result;
        public String base64Image; // Base64 of the raw bytes (no prefix)
    }
}
