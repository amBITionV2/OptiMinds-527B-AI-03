package com.saharsh.Code.editor.Platform.repo;

import com.saharsh.Code.editor.Platform.model.TestCase;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface TestCaseRepository extends JpaRepository<TestCase, Integer> {
    Optional<TestCase> findByQuestionProblemId(int problemId);

}
