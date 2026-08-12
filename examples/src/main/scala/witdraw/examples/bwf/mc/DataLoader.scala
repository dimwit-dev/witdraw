// package witdraw.examples.bwf.mc

// import witdraw.*
// import witdraw.given
// import dimwit.*
// import witdraw.dataframe.DataFrame
// import java.io.File
// import witdraw.dataframe.Col
// import witdraw.dataframe.updated

// object MCDataLoader:

//   type Schema = (
//       Q1: String,
//       Q2: String,
//       Q3: String,
//       Q4: String,
//       Q5: String,
//       Q6: String,
//       Q7: String,
//       Q8: String,
//       Q9: String,
//       Q10: String,
//       Q11: String,
//       Q12: String,
//       Q13: String,
//       Q14: String,
//       Q15: String,
//       Q16: String,
//       Q17: String,
//       Q18: String,
//       Q19: String,
//       Q20: String,
//       Q21: String,
//       Q22: String,
//       Q23: String,
//       Q24: String
//   )

//   case class LongData(
//       studentIndexPerObs: IndexedSeq[Int], // 0-based, length N = J*K
//       itemIndexPerObs: IndexedSeq[Int], // 0-based, length N = J*K
//       isCorrect: IndexedSeq[Boolean], // observed y, length N
//       studentScore: IndexedSeq[Float] // raw score per student, length J
//   ):
//     def nObs = isCorrect.length
//     def nStudents = studentScore.length

//   def loadLongData: LongData =
//     val dataDir = new File(System.getProperty("user.dir"))

//     val correctDf = DataFrame.fromCSV[Schema](
//       new File(dataDir, "data/bwf/MCCorrectAnswers.csv")
//     )
//     val studentDf = DataFrame.fromCSV[Schema](
//       new File(dataDir, "data/bwf/MCStudentAnswers.csv")
//     )

//     // Extract per-question columns: correct answer (1 row) and student answers (drop header)
//     val correctAnswers: IndexedSeq[String] = IndexedSeq(
//       correctDf.col(_.Q1).head,
//       correctDf.col(_.Q2).head,
//       correctDf.col(_.Q3).head,
//       correctDf.col(_.Q4).head,
//       correctDf.col(_.Q5).head,
//       correctDf.col(_.Q6).head,
//       correctDf.col(_.Q7).head,
//       correctDf.col(_.Q8).head,
//       correctDf.col(_.Q9).head,
//       correctDf.col(_.Q10).head,
//       correctDf.col(_.Q11).head,
//       correctDf.col(_.Q12).head,
//       correctDf.col(_.Q13).head,
//       correctDf.col(_.Q14).head,
//       correctDf.col(_.Q15).head,
//       correctDf.col(_.Q16).head,
//       correctDf.col(_.Q17).head,
//       correctDf.col(_.Q18).head,
//       correctDf.col(_.Q19).head,
//       correctDf.col(_.Q20).head,
//       correctDf.col(_.Q21).head,
//       correctDf.col(_.Q22).head,
//       correctDf.col(_.Q23).head,
//       correctDf.col(_.Q24).head
//     )

//     // Student answers per question (drop the CSV header row that fromCSV already exposed)
//     val studentAnswersByQuestion: IndexedSeq[IndexedSeq[String]] = IndexedSeq(
//       studentDf.col(_.Q1).drop(1),
//       studentDf.col(_.Q2).drop(1),
//       studentDf.col(_.Q3).drop(1),
//       studentDf.col(_.Q4).drop(1),
//       studentDf.col(_.Q5).drop(1),
//       studentDf.col(_.Q6).drop(1),
//       studentDf.col(_.Q7).drop(1),
//       studentDf.col(_.Q8).drop(1),
//       studentDf.col(_.Q9).drop(1),
//       studentDf.col(_.Q10).drop(1),
//       studentDf.col(_.Q11).drop(1),
//       studentDf.col(_.Q12).drop(1),
//       studentDf.col(_.Q13).drop(1),
//       studentDf.col(_.Q14).drop(1),
//       studentDf.col(_.Q15).drop(1),
//       studentDf.col(_.Q16).drop(1),
//       studentDf.col(_.Q17).drop(1),
//       studentDf.col(_.Q18).drop(1),
//       studentDf.col(_.Q19).drop(1),
//       studentDf.col(_.Q20).drop(1),
//       studentDf.col(_.Q21).drop(1),
//       studentDf.col(_.Q22).drop(1),
//       studentDf.col(_.Q23).drop(1),
//       studentDf.col(_.Q24).drop(1)
//     )

//     val nStudents = studentAnswersByQuestion.head.length
//     val nItems = 24
//     val N = nStudents * nItems

//     val studentIdx = Array.ofDim[Int](N)
//     val itemIdx = Array.ofDim[Int](N)
//     val y = Array.ofDim[Boolean](N)

//     for j <- 0 until nStudents; k <- 0 until nItems do
//       val i = j * nItems + k
//       studentIdx(i) = j
//       itemIdx(i) = k
//       y(i) = studentAnswersByQuestion(k)(j) == correctAnswers(k)

//     // Raw score per student = number of correct answers
//     val score: IndexedSeq[Float] =
//       (0 until nStudents).map { j =>
//         (0 until nItems)
//           .count(k => studentAnswersByQuestion(k)(j) == correctAnswers(k))
//           .toFloat
//       }

//     LongData(
//       studentIdx.toIndexedSeq,
//       itemIdx.toIndexedSeq,
//       y.toIndexedSeq,
//       score
//     )
