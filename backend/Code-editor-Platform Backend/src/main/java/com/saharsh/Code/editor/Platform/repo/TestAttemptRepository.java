package com.saharsh.Code.editor.Platform.repo;

import com.saharsh.Code.editor.Platform.model.TestAttempt;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface TestAttemptRepository extends JpaRepository<TestAttempt, Integer> {
    List<TestAttempt> findByTest_TestId(Long id);

    List<TestAttempt> findByTestTestId(int id);
    // You can add custom query methods here if needed
}
