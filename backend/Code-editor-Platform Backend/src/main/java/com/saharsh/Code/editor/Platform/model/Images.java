package com.saharsh.Code.editor.Platform.model;


import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.JdbcTypeCode;
import org.hibernate.type.SqlTypes;

import java.time.LocalDateTime;

@Data
@NoArgsConstructor
@Entity
@Table(name = "images")
public class Images {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    private Long userId;

    @Column(nullable = false)
    private Long testId;

    // Map to PostgreSQL bytea
    @JdbcTypeCode(SqlTypes.VARBINARY) // or SqlTypes.BINARY
    @Column(name = "image_data", nullable = false)
    private byte[] imageData;
    @Column(nullable = false)
    private Long time;
    @Column(nullable = false)
    private Boolean isPhone;

    public Images(Long id, Long userId, Long testId, byte[] imageData, Long time,Boolean isPhone) {
        this.id = id;
        this.userId = userId;
        this.testId = testId;
        this.imageData = imageData;
        this.time=time;
        this.isPhone=isPhone;
    }
}
