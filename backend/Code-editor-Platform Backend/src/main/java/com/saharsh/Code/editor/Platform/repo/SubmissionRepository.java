package com.saharsh.Code.editor.Platform.repo;

import com.saharsh.Code.editor.Platform.model.Submission;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface SubmissionRepository extends JpaRepository<Submission, Integer> {
    // You can add custom query methods here if needed
}