# Labelling guideline — v1

One person labels (me). When a case is hard, I write the rule that
decides it here, then apply the rule. I never change a label without
changing or adding a rule.

## Categories

- **billing** — money that moved or should move: charges, refunds,
  invoices, plan prices, failed payments.
- **technical** — the product does not work as designed: crashes,
  errors, bugs, features not loading.
- **account** — who the customer is and how they access things:
  login, password, email address, profile, payment *method* on file.
- **general** — everything else: questions, feedback, feature
  requests, "how do I", messages with no request at all.

## Boundary rules

1. If the customer says a specific amount was wrong, it is billing,
   even if the cause is a bug.
2. A "how do I ..." question about a feature that works is general,
   not technical.
3. **(your call)** Changing the card / payment method on file is
   __________ because __________.
4. **(your call)** When an email raises two issues, the primary
   category is __________ (the one mentioned first? the one involving
   money? the one that blocks the customer?). The other goes in
   `acceptable_categories`.

## Summaries

- One sentence. Names every issue the customer raised.
- Contains nothing the email does not say. "No details given" is a
  valid summary.