package com.saharsh.Code.editor.Platform.repo;


import com.saharsh.Code.editor.Platform.model.TestProblem;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface TestProblemRepository extends JpaRepository<TestProblem, Integer> {
// You can add custom query methods here if needed
}