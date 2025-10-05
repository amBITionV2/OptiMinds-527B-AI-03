package com.saharsh.Code.editor.Platform.repo;

import com.saharsh.Code.editor.Platform.model.Question;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface QuestionRepository extends JpaRepository<Question, Integer> {
    // You can add custom query methods here if needed
}