## Shop Scheduling Problems

Companies face complex processes, conflicts of interest, and high deadline pressure, which demands constant planning and coordination.
This circumstance motivates scheduling research to deal with the general allocation of resources and methods to efficiently complete tasks {cite}`blazewicz2000disjunctive, jaehn2014ablaufplanung, pinedo2005planning`.
Scheduling is a multifaceted field and finds application in many areas.
In computing and IT, processes require assigning CPU cores or distributing large workloads to individual computers.
In logistics, railroad tracks and airport runways are allocated to trains and airplanes, and medical diagnostic equipment gets assigned to patients in hospitals {cite}`blazewicz2000disjunctive, pinedo2005planning`.
The corresponding efficiency criteria for a schedule vary as much as the applications, but most criteria formulate a measure based on time and resource consumption.
This thesis focuses on scheduling in manufacturing, where tasks are assigned to specific machines.
Machine scheduling comes in many different forms.
Some companies manufacture very similar products that all go through the same machines in the same order.
Others are to manufacture many different products whose manufacturing processes are multifarious.
However, one can characterize all scheduling problems arising in this context by three parameters as follows according to {cite}`blazewicz2000disjunctive, jaehn2014ablaufplanung, pinedo2005planning`:

```{admonition} Quote 

{cite}`lange2019solution`, "A machine scheduling problem $\alpha|\beta|\gamma$ is described by the specification of the resources $\alpha$, also called machines, predefined instance characteristics and constraints $\beta$ and an objective function $\gamma$ that is to be minimized "

```



In all shop scheduling problems, a job consists of a finite number of operations, also called tasks, each of which requires processing on a machine.
Particular requirements regarding the sequence of operations manifest in the $\alpha$ parameter.
When all jobs have the same sequence of operations, one speaks of a $\textit{flow shop}$ ($\alpha=Fm$).
If the operations of the jobs require processing in a different order on the machines, this is called a $\textit{job shop}$ ($\alpha=Jm$).
If the sequence of operations within a job is unrestricted, then the problem is an $\textit{open shop}$ ($\alpha=Om$) {cite}`jaehn2014ablaufplanung`.
The $\beta$ parameter enables the definition of additional process properties and constraints.
For example, a constraint could be that a machine can only run in a certain time window.
The objective function $\gamma$ determines which quantity the optimization targets.
For example, finishing all jobs preferably, at the same time could be an objective.
Another one is the minimization of the makespan.
The makespan describes the time from the start of the first operation to the completion of the last operation in a schedule.
Therefore, the makespan is the amount of time needed for the complete processing of all operations.

```{eval-rst}
The above examples only include very few possibilities for the parameters :math:`\alpha`, :math:`\beta`, and :math:`\gamma`.
:cite:t:`jaehn2014ablaufplanung` cover a large variety of possible parameterizations and discuss the resulting problems and solution methods.
```


The job shop problem (JSP) modelling approach offers one of the most flexible forms of description in the context of workshop production is of high practical relevance since customer-specific products play an increasing role in the offerings of many companies {cite}`duffie2017analytical, schuh2019databased`.
`grap-jsp-env` only targets the Job Shop Problem (JSP) without additional constraints.
The makespan, often referred to as $C_{max}$, was chosen as the optimization metric.
This specific form of the problem ($J_m||C_{max}$) is widely used in the literature and is sometimes referred to as the classical Job Shop Problem {cite}`henning2002praktische, gabel2009multi`.
The JSP is formally defined and elaborated in the following section.

